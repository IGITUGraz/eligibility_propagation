from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf

Cell = tf.contrib.rnn.BasicRNNCell
tfe = tf.contrib.eager
# rd = np.random.RandomState(3000)


# PSP on output layer
def exp_convolve(tensor, decay, init=None, axis=1):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        l_shp = len(tensor.get_shape())
        perm = np.arange(l_shp)
        perm[0] = axis
        perm[axis] = 0

        tensor_time_major = tf.transpose(tensor, perm=perm)
        if init is not None:
            assert str(init.get_shape()) == str(tensor_time_major[0].get_shape())  # must be batch x neurons
            initializer = init
        else:
            initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=perm)
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
    def __init__(self, n_in, n_rec, tau=20., thr=.615, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=.16, tag='', rewiring_connectivity=-1,
                 in_neuron_sign=None, rec_neuron_sign=None,
                 stop_gradients=False, w_in_init=None, w_rec_init=None, n_refractory=1, rec=True):
        """
        CustomALIF provides the recurrent tensorflow cell model for implementing LSNNs in combination with
        eligibility propagation (e-prop).
        Cell output is a tuple: z (spikes): n_batch x n_neurons,
                                s (states): n_batch x n_neurons x 2,
                                diag_j (neuron specific jacobian, partial z / partial s): (n_batch x n_neurons x 2 x 2,
                                                                                           n_batch x n_neurons x 2),
                                partials_wrt_biases (partial s / partial input_current for inputs, recurrent spikes):
                                        (n_batch x n_neurons x 2, n_batch x n_neurons x 2)
        UPDATE: This model uses v^{t+1} ~ alpha * v^t + i_t instead of ... + (1 - alpha) * i_t
                it is therefore required to rescale thr, and beta of older version by
                thr = thr_old / (1 - exp(- 1 / tau))
                beta = beta_old * (1 - exp(- 1 / tau_adaptation)) / (1 - exp(- 1 / tau))
        UPDATE: refractory periods are implemented
        :param n_in: number of input neurons
        :param n_rec: number of output neurons
        :param tau: membrane time constant
        :param thr: spike threshold
        :param dt: length of discrete time steps
        :param dtype: data type of tensors
        :param dampening_factor: used in pseudo-derivative
        :param tau_adaptation: time constant of adaptive threshold decay
        :param beta: impact of adapting thresholds
        :param stop_gradients: stop gradients between next cell state and visible states
        :param w_in_init: initial weights for input connections
        :param w_rec_init: initial weights for recurrent connections
        :param n_refractory: number of refractory time steps
        """

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")
        with tf.variable_scope('CustomALIF_' + str(tag)):
            self.n_refractory = n_refractory
            self.tau_adaptation = tau_adaptation
            self.beta = beta
            self.decay_b = np.exp(-dt / tau_adaptation)

            if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
            if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

            tau = tf.cast(tau, dtype=dtype)
            dt = tf.cast(dt, dtype=dtype)
            self.rec = rec

            self.dampening_factor = dampening_factor
            self.stop_gradients = stop_gradients
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
                if rec:
                    init_w_rec_var = w_rec_init if w_rec_init is not None else \
                        (rd.randn(n_rec, n_rec) / np.sqrt(n_rec)).astype(np.float32)
                    self.w_rec_var = tf.get_variable('RecurrentWeight', initializer=init_w_rec_var, dtype=dtype)
                    self.w_rec_val = self.w_rec_var

                    self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

                    # Disconnect autotapse
                    self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),self.w_rec_val)

                    dw_val_dw_var_rec = np.ones((self._num_units,self._num_units)) - np.diag(np.ones(self._num_units))
            dw_val_dw_var_in = np.ones((n_in,self._num_units))

            self.dw_val_dw_var = [dw_val_dw_var_in, dw_val_dw_var_rec] if rec else [dw_val_dw_var_in,]

            self.variable_list = [self.w_in_var, self.w_rec_var] if rec else [self.w_in_var,]
            self.built = True


    @property
    def state_size(self):
        return CustomALIFStateTuple(s=tf.TensorShape((self.n_rec, 2)), z=self.n_rec, r=self.n_rec)

    def set_weights(self, w_in, w_rec):
        recurrent_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
        w_rec_rank = len(w_rec.get_shape().as_list())
        if w_rec_rank == 3:
            n_batch = tf.shape(w_rec)[0]
            recurrent_disconnect_mask = tf.tile(recurrent_disconnect_mask[None, ...], (n_batch, 1, 1))

        self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(w_rec), w_rec)
        self.w_in_val = w_in

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

        new_b = self.decay_b * b + old_z

        if len(self.w_in_val.get_shape().as_list()) == 3:
            i_in = tf.einsum('bi,bij->bj', inputs, self.w_in_val)
        else:
            i_in = tf.matmul(inputs, self.w_in_val)
        if self.rec:
            if len(self.w_rec_val.get_shape().as_list()) == 3:
                i_rec = tf.einsum('bi,bij->bj', z, self.w_rec_val)
            else:
                i_rec = tf.matmul(z, self.w_rec_val)
            i_t = i_in + i_rec
        else:
            i_t = i_in

        I_reset = z * self.thr * self.dt

        new_v = decay * v + i_t - I_reset

        # Spike generation
        is_refractory = tf.greater(state.r, .1)
        zeros_like_spikes = tf.zeros_like(state.z)
        new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                 0., float(self.n_refractory))
        new_s = tf.stack((new_v, new_b), axis=-1)

        def safe_grad(y, x):
            g = tf.gradients(y, x)[0]
            if g is None:
                g = tf.zeros_like(x)
            return g

        dnew_v_ds = tf.gradients(new_v, s, name='dnew_v_ds')[0]
        dnew_b_ds = tf.gradients(new_b, s, name='dnew_b_ds')[0]
        dnew_s_ds = tf.stack((dnew_v_ds, dnew_b_ds), 2, name='dnew_s_ds')

        dnew_z_dnew_v = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_v))
        dnew_z_dnew_b = tf.where(is_refractory, zeros_like_spikes, safe_grad(new_z, new_b))
        dnew_z_dnew_s = tf.stack((dnew_z_dnew_v, dnew_z_dnew_b), axis=-1)

        diagonal_jacobian = [dnew_s_ds, dnew_z_dnew_s]

        # "in_weights, rec_weights"
        # ds_dW_bias: 2 x n_rec
        dnew_v_di = safe_grad(new_v,i_t)
        dnew_b_di = safe_grad(new_b,i_t)
        dnew_s_di = tf.stack([dnew_v_di,dnew_b_di], axis=-1)

        partials_wrt_biases = [dnew_s_di, dnew_s_di]

        new_state = CustomALIFStateTuple(s=new_s, z=new_z, r=new_r)
        return [new_z, new_s, diagonal_jacobian, partials_wrt_biases], new_state

