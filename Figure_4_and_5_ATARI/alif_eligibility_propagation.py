# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass

from collections import namedtuple

import numpy as np
import tensorflow as tf

Cell = tf.contrib.rnn.BasicRNNCell


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


CustomALIFStateTuple = namedtuple('CustomALIFStateTuple', ('s', 'z', 'r'))


class CustomALIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03, dt=1., dtype=tf.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=1.6, stop_gradients=False, w_in_init=None, w_rec_init=None,
                 n_refractory=5, w_scale=1.):
        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)

        if np.isscalar(tau):
            tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr):
            thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

        tau = tf.cast(tau, dtype=dtype)
        dt = tf.cast(dt, dtype=dtype)

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
                (np.random.randn(n_in, n_rec) / np.sqrt(n_in)).astype(np.float32)
            init_w_in_var *= w_scale
            self.w_in_var = tf.get_variable("InputWeight", initializer=init_w_in_var, dtype=dtype)
            self.w_in_val = self.w_in_var

        with tf.variable_scope('RecWeights'):
            init_w_rec_var = w_rec_init if w_rec_init is not None else \
                (np.random.randn(n_rec, n_rec) / np.sqrt(n_rec)).astype(np.float32)
            init_w_rec_var *= w_scale
            self.w_rec_var = tf.get_variable('RecurrentWeight', initializer=init_w_rec_var, dtype=dtype)
            self.w_rec_val = self.w_rec_var

            self.recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

            self.w_rec_val = tf.where(self.recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),self.w_rec_val)  # Disconnect autotapse

        dw_val_dw_var_in = np.ones((n_in,self._num_units))
        dw_val_dw_var_rec = np.ones((self._num_units,self._num_units)) - np.diag(np.ones(self._num_units))
        self.dw_val_dw_var = [dw_val_dw_var_in,dw_val_dw_var_rec]

        self.variable_list = [self.w_in_var,self.w_rec_var]
        self.built = True

    @property
    def state_size(self):
        return CustomALIFStateTuple(s=tf.TensorShape((self.n_rec, 2)), z=self.n_rec, r=self.n_rec)

    @property
    def output_size(self):
        return [self.n_rec, tf.TensorShape((self.n_rec, 2)), tf.TensorShape((1,)), tf.TensorShape((1,))]

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        s0 = tf.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        return CustomALIFStateTuple(s=s0, z=z0, r=r0)

    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = spike_function(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        decay = self._decay

        z = state.z
        s = state.s
        v, b = s[..., 0], s[..., 1]

        is_refractory = tf.greater(state.r, .1)
        zeros_like_spikes = tf.zeros_like(state.z)
        old_z = z

        if self.stop_gradients:
            z = tf.stop_gradient(z)

        new_b = self.decay_b * b + old_z

        i_t = tf.matmul(inputs, self.w_in_val) + tf.matmul(z, self.w_rec_val)
        I_reset = z * self.thr * self.dt
        new_v = decay * v + i_t - I_reset

        # Spike generation
        new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = tf.stop_gradient(tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                                  0., float(self.n_refractory)))
        new_s = tf.stack((new_v, new_b), axis=-1)

        new_state = CustomALIFStateTuple(s=new_s, z=new_z, r=new_r)
        return [new_z, new_s, tf.zeros_like(new_z[:, :1]), tf.zeros_like(new_z[:, :1])], new_state


class CustomALIFWithReset(CustomALIF):
    def __call__(self, inputs, state, scope=None, dtype=tf.float32):
        state_reset = inputs[..., -1] > .5
        inputs = inputs[..., :-1]
        zero_state = self.zero_state(tf.shape(inputs)[0], tf.float32)
        state_reset_z = tf.tile(state_reset[..., None], (1, zero_state.z.get_shape()[1]))
        state_reset_s = tf.tile(state_reset[..., None, None], (1, zero_state.z.get_shape()[1], 2))
        state_z = tf.where(state_reset_z, zero_state.z, state.z)
        state_s = tf.where(state_reset_s, zero_state.s, state.s)
        state_r = tf.where(state_reset_z, zero_state.r, state.r)
        state = CustomALIFStateTuple(z=state_z, s=state_s, r=state_r)

        output, new_state = super().__call__(inputs, state, scope=scope, dtype=dtype)

        new_z, new_s, _unused_1, _unused_2 = output

        return [new_z, new_s, _unused_1, _unused_2], new_state
