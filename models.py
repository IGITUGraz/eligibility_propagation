import tensorflow as tf
import numpy as np
from collections import namedtuple

Cell = tf.contrib.rnn.BasicRNNCell
LightLIFStateTuple = namedtuple('LightLIFStateTuple', ('v', 'z'))


def pseudo_derivative(v_scaled, dampening_factor):
    return tf.maximum(1 - tf.abs(v_scaled), 0) * dampening_factor


@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


class LightLIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.6, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 stop_z_gradients=False):

        self.dampening_factor = dampening_factor
        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype
        self.stop_z_gradients = stop_z_gradients

        self._num_units = self.n_rec

        self.tau = tf.constant(tau, dtype=dtype)
        self._decay = tf.exp(-dt / self.tau)
        self.thr = thr

        with tf.variable_scope('InputWeights'):
            self.w_in_var = tf.Variable(np.random.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype)
            self.w_in_val = tf.identity(self.w_in_var)

        with tf.variable_scope('RecWeights'):
            self.w_rec_var = tf.Variable(np.random.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype)
            self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_var),
                                      self.w_rec_var)  # Disconnect autotapse

    @property
    def state_size(self):
        return LightLIFStateTuple(v=self.n_rec, z=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, self.n_rec]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return LightLIFStateTuple(v=v0, z=z0)

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        thr = self.thr
        z = state.z
        v = state.v
        decay = self._decay

        if self.stop_z_gradients:
            z_stop = tf.stop_gradient(z)
        else:
            z_stop = z

        # update the voltage
        i_t = tf.matmul(inputs, self.w_in_val) + tf.matmul(z_stop, self.w_rec_val)
        I_reset = z_stop * self.thr * self.dt
        new_v = decay * v + i_t - I_reset

        # Spike generation
        v_scaled = (new_v - thr) / thr
        new_z = SpikeFunction(v_scaled, self.dampening_factor)
        new_z = new_z * 1 / self.dt

        new_state = LightLIFStateTuple(v=new_v, z=new_z)
        return [new_z, new_v], new_state


def exp_convolve(tensor, decay):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
        r_shp = range(len(tensor.get_shape()))
        transpose_perm = [1, 0] + list(r_shp)[2:]

        tensor_time_major = tf.transpose(tensor, perm=transpose_perm)
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=transpose_perm)
    return filtered_tensor


def shift_by_one_time_step(tensor, initializer=None):
    with tf.name_scope('TimeShift'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
        r_shp = range(len(tensor.get_shape()))
        transpose_perm = [1, 0] + list(r_shp)[2:]

        tensor_time_major = tf.transpose(tensor, perm=transpose_perm)
        if initializer is None:
            initializer = tf.zeros_like(tensor_time_major[0])

        shifted_tensor = tf.concat([initializer[None, :, :], tensor_time_major[:-1]], axis=0)

        shifted_tensor = tf.transpose(shifted_tensor, perm=transpose_perm)
    return shifted_tensor
