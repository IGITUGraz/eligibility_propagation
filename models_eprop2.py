import tensorflow as tf
import numpy as np
from collections import namedtuple

Cell = tf.contrib.rnn.BasicRNNCell


def pseudo_derivative(v_scaled, dampening_factor):
    """
    Define the pseudo derivative used to derive through spikes.
    :param v_scaled: scaled version of the voltage being 0 at threshold and -1 at rest
    :param dampening_factor: parameter that stabilizes learning
    :return:
    """
    return tf.maximum(1 - tf.abs(v_scaled), 0) * dampening_factor


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    """
    The tensorflow function which is defined as a Heaviside function (to compute the spikes),
    but with a gradient defined with the pseudo derivative.
    :param v_scaled: scaled version of the voltage being -1 at rest and 0 at the threshold
    :param dampening_factor: parameter to stabilize learning
    :return: the spike tensor
    """
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


class LIF(Cell):
    State = namedtuple('LIFStateTuple', ('v', 'z', 'r'))

    def __init__(self, n_in, n_rec, tau=20., thr=0.4, dt=1., dtype=tf.float32, dampening_factor=0.3, n_refractory=5):

        self.dampening_factor = dampening_factor
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self.decay = tf.exp(-dt / tau)
        self.thr = thr
        self.stop_gradient = False
        self.n_refractory = n_refractory

        with tf.variable_scope('input_weights'):
            init_w_in_var = np.random.randn(n_in, n_rec) / np.sqrt(n_in)
            init_w_in_var = tf.cast(init_w_in_var, dtype)
            self.w_in_var = tf.get_variable('w', initializer=init_w_in_var)
            self.w_in_val = self.w_in_var

        with tf.variable_scope('recurrent_weights'):
            init_w_rec_var = np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)
            init_w_rec_var = tf.cast(init_w_rec_var, dtype)
            self.w_rec_var = tf.get_variable('w', initializer=init_w_rec_var)
            self.w_rec_val = self.w_rec_var

            # disconnect autapse
            self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)

    @property
    def state_size(self):
        return LIF.State(v=self.n_rec, z=self.n_rec, r=self.n_rec)

    @property
    def output_size(self):
        return self.n_rec, self.n_rec

    def zero_state(self, batch_size, dtype):
        v0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, self.n_rec), dtype=dtype)

        return LIF.State(v=v0, z=z0, r=r0)

    def set_weights(self, w_in, w_rec):
        recurrent_disconnect_mask = np.diag(np.ones(self.n_rec, dtype=bool))
        w_rec_rank = len(w_rec.get_shape().as_list())
        if w_rec_rank == 3:
            n_batch = tf.shape(w_rec)[0]
            recurrent_disconnect_mask = tf.tile(recurrent_disconnect_mask[None, ...], (n_batch, 1, 1))

        self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(w_rec), w_rec)
        self.w_in_val = w_in

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        w_in_has_batch_dim = len(self.w_in_val.get_shape().as_list()) == 3
        w_rec_has_batch_dim = len(self.w_rec_val.get_shape().as_list()) == 3

        previous_z = state.z

        if w_in_has_batch_dim:
            i_in = tf.einsum('bi,bij->bj', inputs, self.w_in_val)
        else:
            i_in = tf.matmul(inputs, self.w_in_val)

        if w_rec_has_batch_dim:
            i_rec = tf.einsum('bi,bij->bj', previous_z, self.w_rec_val)
        else:
            i_rec = tf.matmul(previous_z, self.w_rec_val)

        i_reset = previous_z * self.thr * self.dt
        new_v = self.decay * state.v + (i_in + i_rec) - i_reset

        # Spike generation
        v_scaled = (new_v - self.thr) / self.thr
        new_z = spike_function(v_scaled, self.dampening_factor)
        new_z = new_z * 1 / self.dt

        # check if refractory
        is_refractory = tf.greater(state.r, .1)
        zeros_like_spikes = tf.zeros_like(state.z)
        new_z = tf.where(is_refractory, zeros_like_spikes, new_z)
        new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                 0., float(self.n_refractory))

        new_state = LIF.State(v=new_v, z=new_z, r=new_r)

        return (new_z, new_v), new_state


def spike_encode(input_component, minn, maxx, n_input_code=100, max_rate_hz=200, dt=1, n_dt_per_step=None):
    """
    Population-rate encode analog values

    :param input_component: tensor of analog values
    :param minn: minimum value that this population can encode
    :param maxx: maximum value that this population can encode
    :param n_input_code: number of neurons that encode this value
    :param max_rate_hz: maximum rate for tuned neurons
    :param dt:
    :param n_dt_per_step: number of time steps a single analog value is encoded
    :return: A spike tensor that encodes the analog values
    """
    if 110 < n_input_code < 210:  # 100
        factor = 20
    elif 90 < n_input_code < 110:  # 100
        factor = 10
    elif 15 < n_input_code < 25:  # 20
        factor = 4
    else:
        factor = 2

    sigma_tuning = (maxx - minn) / n_input_code * factor
    mean_line = tf.cast(tf.linspace(minn - 2. * sigma_tuning, maxx + 2. * sigma_tuning, n_input_code), tf.float32)
    max_rate = max_rate_hz / 1000
    max_prob = max_rate * dt

    step_neuron_firing_prob = max_prob * tf.exp(-(mean_line[None, None, :] - input_component[..., None]) ** 2 /
                                                (2 * sigma_tuning ** 2))

    if n_dt_per_step is not None:
        spike_code = tf.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample(n_dt_per_step)
        dims = len(spike_code.get_shape())
        r = list(range(dims))
        spike_code = tf.transpose(spike_code, r[1:-1] + [0, r[-1]])
    else:
        spike_code = tf.distributions.Bernoulli(probs=step_neuron_firing_prob, dtype=tf.bool).sample()

    spike_code = tf.cast(spike_code, tf.float32)
    return spike_code


def exp_convolve(tensor, decay):
    with tf.name_scope('exp_convolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


