from collections import namedtuple
from tensorflow.python.util import nest

import numpy as np
import tensorflow as tf
import numpy.random as rd

Cell = tf.contrib.rnn.BasicRNNCell
tfe = tf.contrib.eager

def exp_convolve(tensor, decay):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


CustomALIFStateTuple = namedtuple('CustomALIFStateTuple', ('s', 'z', 'r'))


class CustomALIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=1.6,
                 stop_gradients=False, no_elig=False, w_in_init=None, w_rec_init=None, n_refractory=1):

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)

        if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

        self.dampening_factor = dampening_factor
        self.stop_gradients = stop_gradients
        self.no_elig = no_elig
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = tf.exp(-dt / tau)
        self.thr = thr

        with tf.variable_scope('InputWeights'):
            # Input weights
            init_w_in_var = w_in_init if w_in_init is not None else \
                (rd.randn(n_in, n_rec) / np.sqrt(n_in)).astype(np.float32)
            self.w_in_var = tf.get_variable("InputWeight", initializer=init_w_in_var, dtype=dtype)
            self.w_in_val = self.w_in_var

        with tf.variable_scope('RecWeights'):
            init_w_rec_var = w_rec_init if w_rec_init is not None else \
                (rd.randn(n_rec, n_rec) / np.sqrt(n_rec)).astype(np.float32)
            self.w_rec_var = tf.get_variable('RecurrentWeight', initializer=init_w_rec_var, dtype=dtype)
            self.w_rec_val = self.w_rec_var
            self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),
                                      self.w_rec_val)  # Disconnect autotapse

        # TODO: Think about how to handle that more cleverly for the case with rewiring with sparse tensors
        dw_val_dw_var_in = np.ones((n_in, self._num_units))
        dw_val_dw_var_rec = np.ones((self._num_units, self._num_units)) - np.diag(np.ones(self._num_units))
        self.dw_val_dw_var = [dw_val_dw_var_in, dw_val_dw_var_rec]

        self.variable_list = [self.w_in_var, self.w_rec_var]
        self.built = True

    @property
    def state_size(self):
        return CustomALIFStateTuple(s=tf.TensorShape((self.n_rec, 2)), z=self.n_rec, r=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, tf.TensorShape((self.n_rec, 2)),
                [tf.TensorShape((self.n_rec, 2, 2)), tf.TensorShape((self.n_rec, 2))],
                [tf.TensorShape((self.n_rec, 2))] * 2]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        s0 = tf.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return CustomALIFStateTuple(s=s0, z=z0, r=r0)

    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = SpikeFunction(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        decay = self._decay

        z = state.z
        s = state.s
        v, b = s[..., 0], s[..., 1]

        old_z = self.compute_z(v, b)

        if self.stop_gradients:
            z = tf.stop_gradient(z)
        if self.no_elig:
            v = tf.stop_gradient(v)

        new_b = self.decay_b * b + old_z

        i_t = tf.matmul(inputs, self.w_in_val) + tf.matmul(z, self.w_rec_val)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + i_t - I_reset

        # Spike generation
        is_refractory = tf.greater(state.r, .1)
        zeros_like_spikes = tf.zeros_like(state.z)
        new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = tf.stop_gradient(
            tf.clip_by_value(state.r + self.n_refractory * new_z - 1, 0., float(self.n_refractory)))
        new_s = tf.stack((new_v, new_b), axis=-1)

        def safe_grad(y, x):
            g = tf.gradients(y, x)[0]
            if g is None:
                g = tf.zeros_like(x)
            return g

        dnew_v_ds = tf.identity(safe_grad(new_v, s), name='dnew_v_ds')
        dnew_b_ds = tf.identity(safe_grad(new_b, s), name='dnew_b_ds')
        dnew_s_ds = tf.stack((dnew_v_ds, dnew_b_ds), 2, name='dnew_s_ds')

        dnew_z_dnew_v = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_v))
        dnew_z_dnew_b = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_b))
        dnew_z_dnew_s = tf.stack((dnew_z_dnew_v, dnew_z_dnew_b), axis=-1)

        diagonal_jacobian = [dnew_s_ds, dnew_z_dnew_s]

        # "in_weights, rec_weights"
        # ds_dW_bias: 2 x n_rec
        dnew_v_di = safe_grad(new_v, i_t)
        dnew_b_di = safe_grad(new_b, i_t)
        dnew_s_di = tf.stack([dnew_v_di, dnew_b_di], axis=-1)

        partials_wrt_biases = [dnew_s_di, dnew_s_di]

        new_state = CustomALIFStateTuple(s=new_s, z=new_z, r=new_r)
        return [new_z, new_s, diagonal_jacobian, partials_wrt_biases], new_state

    def compute_eligibility_traces(self, v_scaled, z_pre, z_post, is_rec):

        n_neurons = tf.shape(z_post)[2]
        rho = self.decay_b
        beta = self.beta
        alpha = self._decay
        n_ref = self.n_refractory

        # everything should be time major
        z_pre = tf.transpose(z_pre, perm=[1, 0, 2])
        v_scaled = tf.transpose(v_scaled, perm=[1, 0, 2])
        z_post = tf.transpose(z_post, perm=[1, 0, 2])

        psi_no_ref = self.dampening_factor / self.thr * tf.maximum(0., 1. - np.abs(v_scaled))

        update_refractory = lambda refractory_count, z_post: tf.where(z_post > 0,
                                                                      tf.ones_like(refractory_count) * (n_ref - 1),
                                                                      tf.maximum(0, refractory_count - 1))
        refractory_count_init = tf.zeros_like(z_post[0], dtype=tf.int32)
        refractory_count = tf.scan(update_refractory, z_post[:-1], initializer=refractory_count_init, )
        refractory_count = tf.concat([[refractory_count_init], refractory_count], axis=0)

        is_refractory = refractory_count > 0
        psi = tf.where(is_refractory, tf.zeros_like(psi_no_ref), psi_no_ref)

        update_epsilon_v = lambda epsilon_v, z_pre: alpha[None, None, :] * epsilon_v + z_pre[:, :, None]
        epsilon_v_zero = tf.ones((1, 1, n_neurons)) * z_pre[0][:, :, None]
        epsilon_v = tf.scan(update_epsilon_v, z_pre[1:], initializer=epsilon_v_zero, )
        epsilon_v = tf.concat([[epsilon_v_zero], epsilon_v], axis=0)

        update_epsilon_a = lambda epsilon_a, elems: (rho - beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][
                                                                                                          :, None, :] * \
                                                    elems['epsi']

        epsilon_a_zero = tf.zeros_like(epsilon_v[0])
        epsilon_a = tf.scan(fn=update_epsilon_a,
                            elems={'psi': psi_no_ref[:-1], 'epsi': epsilon_v[:-1], },
                            initializer=epsilon_a_zero, )
        epsilon_a = tf.concat([[epsilon_a_zero], epsilon_a], axis=0)

        e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

        # everything should be time major
        e_trace = tf.transpose(e_trace, perm=[1, 0, 2, 3])
        epsilon_v = tf.transpose(epsilon_v, perm=[1, 0, 2, 3])
        epsilon_a = tf.transpose(epsilon_a, perm=[1, 0, 2, 3])
        psi = tf.transpose(psi, perm=[1, 0, 2])

        if is_rec:
            identity_diag = tf.eye(n_neurons)[None, None, :, :]
            e_trace -= identity_diag * e_trace
            epsilon_v -= identity_diag * epsilon_v
            epsilon_a -= identity_diag * epsilon_a

        return e_trace, epsilon_v, epsilon_a, psi

    def compute_loss_gradient(self, learning_signal, z_pre, z_post, v_post, b_post, decay_out=None,
                              zero_on_diagonal=None):
        thr_post = self.thr + self.beta * b_post
        v_scaled = (v_post - thr_post) / self.thr

        e_trace, epsilon_v, epsilon_a, _ = self.compute_eligibility_traces(v_scaled, z_pre, z_post, zero_on_diagonal)

        if decay_out is not None:
            e_trace = tf.transpose(e_trace, perm=[1, 0, 2, 3])
            filtering = lambda filtered_e, e: decay_out * filtered_e + (1 - decay_out) * e
            filtered_e_zero = tf.zeros_like(e_trace[0])
            filtered_e = tf.scan(filtering, e_trace[:-1], initializer=filtered_e_zero)
            filtered_e = tf.concat([[filtered_e_zero], filtered_e], axis=0)

            filtered_e = tf.transpose(filtered_e, perm=[1, 0, 2, 3])
            e_trace = filtered_e

        gradient = tf.einsum('btj,btij->ij', learning_signal, e_trace)

        return gradient, e_trace, epsilon_v, epsilon_a


def tile_matrix_to_match(matrix, match_matrix):
    n_batch_dims = len(match_matrix.get_shape()) - len(matrix.get_shape())
    for _ in range(n_batch_dims):
        matrix = matrix[None, ...]

    tile_dims = tf.shape(match_matrix)
    tile_dims = tf.concat((tile_dims[:-2], [1, 1]), -1)
    matrix = tf.tile(matrix, tile_dims, name='tile_matrix_to_match')
    return matrix


def compute_eligibility_traces_and_gradients_alif(cell, inputs,
                                                  outputs,
                                                  learning_signals,
                                                  initial_eligibility_traces=None,
                                                  initial_state=None,
                                                  synthetic_gradient=None):
    visibles, _, diagonal_jacobians, partials_wrt_biases = outputs
    dnew_s_ds, dnew_z_dnew_s = diagonal_jacobians

    with tf.name_scope('EligibilityTraceComputer'):
        # preparing useful values
        previous_visibles = tf.concat([initial_state.z[:, None], visibles[:, :-1]], axis=1)
        in_rec = [inputs, previous_visibles]

        # Accumulate the history factors derivative of the mapping c(t) -> c(t+1)
        jacobi_identity = tile_matrix_to_match(tf.eye(tf.shape(dnew_s_ds)[-1]), dnew_s_ds[:, 0, ...])[:, None]
        m = tf.transpose(tf.concat([dnew_s_ds, jacobi_identity], axis=1), (1, 0, 2, 3, 4))

        def cum_backward(_previous, _inputs):
            return tf.einsum('bnsk,bnkj->bnsj', _previous, _inputs)

        initializer = tf.tile(tf.eye(2)[None, None, ...], (tf.shape(inputs)[0], cell.n_rec, 1, 1))
        m_t_to_tau = tf.scan(cum_backward, m, initializer=initializer, reverse=True)
        m_t_to_tau = tf.transpose(m_t_to_tau, (1, 0, 2, 3, 4))

        full_history_factor = m_t_to_tau[:, 0]
        m_t_to_tau = m_t_to_tau[:, 1:]

        with tf.name_scope('CEligComputer'):
            # Compute the eligibility traces for c
            dfinalc_db_t = [tf.einsum('btjs,btjks->btjk', p_t, m_t_to_tau) for p_t in partials_wrt_biases]
            # implements D^{t'-1} ... D^{t} dnew_s_dbiases outer_product inputs
            #             |----------------------------| <- dfinalc_db_t
            #                                                             |----| <- in_rec
            # two list elements: input_weights, recurrent weights
            # final shape: n_batch x n_states x n_pre x n_post
            new_eligibility_traces = [tf.einsum('bti,btjs->bijs', u, dnew_s_db) * dw[None, :, :, None]
                                      for u, dnew_s_db, dw in zip(in_rec, dfinalc_db_t, cell.dw_val_dw_var)]

            old_term_eligibility_traces = [tf.einsum('bijs,bjks->bijk', e, full_history_factor) for e in
                                           initial_eligibility_traces]

            final_eligibility_traces = [e_new + e_old for e_new, e_old in
                                        zip(new_eligibility_traces, old_term_eligibility_traces)]

    def cum_forward(_previous, _inputs):
        return tf.einsum('bnsk,bnkj->bnsj', _inputs, _previous)

    with tf.name_scope('GradientComputer'):
        dnew_s_ds_time_major = tf.transpose(dnew_s_ds, (1, 0, 2, 3, 4))
        history_factor_forward_accumulated_time_major = tf.scan(cum_forward, dnew_s_ds_time_major)
        history_factor_forward_accumulated = \
            tf.transpose(history_factor_forward_accumulated_time_major, (1, 0, 2, 3, 4))

        learning_signals_wrt_s = tf.einsum('btj,btjs->btjs', learning_signals, dnew_z_dnew_s)
        backpropagated_learning_signals = tf.einsum('btjsk,btjs->bjk',
                                                    history_factor_forward_accumulated,
                                                    learning_signals_wrt_s)

        if synthetic_gradient is not None:
            backpropagated_synthetic_gradient_partial = [tf.einsum('btjk,bjk->btj', dnew_s_db, synthetic_gradient.s) for
                                                         dnew_s_db in dfinalc_db_t]
            synthetic_gradient_correction = [tf.einsum('btj,bti->ij', sg_partial, u) * dw for sg_partial, u, dw in
                                             zip(backpropagated_synthetic_gradient_partial, in_rec, cell.dw_val_dw_var)]
            gradients_wrt_weights = [tf.einsum('bjs,bijs->ij', backpropagated_learning_signals, e) - sg_correction
                                     for e, sg_correction in
                                     zip(initial_eligibility_traces, synthetic_gradient_correction)]
        else:
            gradients_wrt_weights = [tf.einsum('bjs,bijs->ij', backpropagated_learning_signals, e)
                                     for e in initial_eligibility_traces]

    return final_eligibility_traces, gradients_wrt_weights


def fast_forward_prop_with_alif(cell, inputs, learning_signal_function, dtype=tf.float32,
                                partial_error_function=None,
                                initial_eligibility_traces=None,
                                initial_state=None,
                                return_all_eligibility_traces=False):
    assert partial_error_function is None, 'Not yet implemented'

    batch_size = tf.shape(inputs)[0]
    n_time = tf.shape(inputs)[1]

    assert isinstance(cell, CustomALIF)

    if not cell.built:
        # this is a proxy for building the cell and creating the variables
        cell(inputs[:, 0, :], cell.zero_state(batch_size, tf.float32))

    var_list = cell.variable_list

    if initial_state is None:
        initial_state = cell.zero_state(batch_size, dtype)

    if initial_eligibility_traces is None:
        initial_eligibility_traces = [tf.zeros(shape=tf.concat([[batch_size], tf.shape(v), [2]], axis=0))
                                      for v in var_list]
    if return_all_eligibility_traces:
        initial_elibibility_traces_arrays = nest.map_structure(
            lambda x: tf.TensorArray(tf.float32, element_shape=x.get_shape(), size=n_time),
            initial_eligibility_traces)
    else:
        initial_elibibility_traces_arrays = tf.zeros(())

    cum_var_gradients = [tf.zeros_like(v, name='CumGradients') for v in var_list]

    output_sizes_flat = nest.flatten(cell.output_size)
    append_none = lambda s: [None, s] if isinstance(s, int) else [None] + list(s)
    output_arrays = [tf.TensorArray(dtype=dtype, size=n_time, element_shape=append_none(size), name='OutputArray')
                     for size in output_sizes_flat]

    loop_vars = [0, initial_state, initial_eligibility_traces, cum_var_gradients, output_arrays,
                 initial_elibibility_traces_arrays]

    def loop_condition(t, state, eligibility_traces, cum_weight_updates, output_arrays, elibigility_traces_arrays):
        return t < n_time

    def loop_body(t, state, eligibility_traces, cum_var_gradients, output_arrays, eligibility_traces_arrays):
        input_t = inputs[:, t]

        # Simulate network dynamics
        new_output, new_state = cell(input_t, state)
        new_z, new_visibles, diagonal_jacobian, partials_dnew_state_wrt_b = new_output
        dnew_s_ds, dnew_z_dnew_s = diagonal_jacobian

        new_output_flat = nest.flatten(new_output)
        new_output_arrays = [arr.write(t, new_o) for arr, new_o in zip(output_arrays, new_output_flat)]

        with tf.name_scope('LearningSignalComputer'):
            learning_signals = learning_signal_function(t, new_visibles)

            dl_dnew_s = tf.einsum('bjs,bj->bjs', dnew_z_dnew_s, learning_signals)

        with tf.name_scope('PartialComputer'):
            # Compute partial gradients wrt to parameters
            partials_dnew_state_wrt_w = [tf.einsum('bjs,bi->bijs', dnew_s_db, u) for dnew_s_db, u in
                                         zip(partials_dnew_state_wrt_b, [input_t, state.z])]

            partials_dnew_state_wrt_w = [p * dw[None, :, :, None]
                                         for p, dw in zip(partials_dnew_state_wrt_w, cell.dw_val_dw_var)]

        def update_eligibility_trace_and_weight_updapte(dw, e, var, new_partial_ds_dvar):
            with tf.name_scope('EligibilityTraceAndWeightUpdateComputer'):
                if not var in var_list:
                    return dw, e

                e_rank = len(e.get_shape())
                assert e_rank == 4

                new_e = tf.einsum('bijs,bjks->bijk', e, dnew_s_ds)
                new_e += new_partial_ds_dvar

                g = tf.zeros_like(var)
                if new_e is not None and dl_dnew_s is not None:
                    g += tf.einsum('bijs,bjs->ij', new_e, dl_dnew_s)

                g.set_shape(dw.get_shape())

            return dw + g, new_e

        new_es_and_dws = [update_eligibility_trace_and_weight_updapte(dw, e, var, new_partial)
                          for dw, e, var, new_partial in
                          zip(cum_var_gradients, eligibility_traces, var_list, partials_dnew_state_wrt_w)]

        new_cum_var_gradients = [dw for dw, e, in new_es_and_dws]
        new_eligibility_traces = [e for dw, e, in new_es_and_dws]

        if return_all_eligibility_traces:
            new_eligibility_traces_arrays = []
            if return_all_eligibility_traces:
                for _e, _ta in zip(new_eligibility_traces, eligibility_traces_arrays):
                    new_eligibility_traces_arrays.append(_ta.write(t, _e))
        else:
            new_eligibility_traces_arrays = eligibility_traces_arrays

        new_t = t + 1

        return new_t, new_state, new_eligibility_traces, new_cum_var_gradients, new_output_arrays, \
               new_eligibility_traces_arrays

    loop_vars = tf.while_loop(cond=loop_condition,
                              body=loop_body,
                              loop_vars=loop_vars,
                              parallel_iterations=1)

    t_end, final_state, final_eligibility_traces, cum_var_gradients, output_arrays, all_eligibility_traces_arrays = \
        loop_vars

    if return_all_eligibility_traces:
        all_eligibility_traces_arrays = \
            nest.map_structure(lambda x: tf.transpose(x.stack(), (1, 0, 2, 3, 4)), all_eligibility_traces_arrays)

    def transpose_and_stack(arr):
        arr = arr.stack()
        if len(arr.get_shape()) == 3:
            return tf.transpose(arr, perm=[1, 0, 2])
        elif len(arr.get_shape()) == 4:
            return tf.transpose(arr, perm=[1, 0, 2, 3])
        elif len(arr.get_shape()) == 5:
            return tf.transpose(arr, perm=[1, 0, 2, 3, 4])
        else:
            raise NotImplementedError('unkown shape ' + str(arr.get_shape()))

    outputs_flat = [transpose_and_stack(arr) for arr in output_arrays]
    outputs = nest.pack_sequence_as(cell.output_size, outputs_flat)

    if return_all_eligibility_traces:
        return final_state, final_eligibility_traces, cum_var_gradients, outputs, all_eligibility_traces_arrays
    return final_state, final_eligibility_traces, cum_var_gradients, outputs  # , learning_signal_targets
