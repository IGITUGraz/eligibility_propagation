# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass

import collections
import environments
import sonnet as snt
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from alif_eligibility_propagation import CustomALIFWithReset, spike_function
from util import to_bool, switch_time_and_batch_dimension, exp_convolve


AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


def lif_dynamic(v, i, decay, v_th, dampening_factor=.3):
    old_z = spike_function((v - v_th) / v_th, dampening_factor)
    new_v = decay * v + i - old_z * v_th
    new_z = spike_function((new_v - v_th) / v_th, dampening_factor)
    return new_v, new_z


class SpikingCNN(tf.nn.rnn_cell.RNNCell):
    def __init__(self, n_kernel_1=8, n_filter_1=16, stride_1=4, n_kernel_2=4, n_filter_2=32, stride_2=2, ba=False,
                 avg_ba=False, ba_config=None, tau=1, thr=1., avg_pool_1_stride=4, avg_pool_1_k=8, avg_pool_2_stride=2,
                 avg_pool_2_k=4):
        super().__init__()
        self.decay = np.exp(-1 / tau)
        self.v_th = thr
        self.n_filters_1 = n_filter_1
        self.n_filters_2 = n_filter_2
        self.ba = ba
        self.avg_ba = avg_ba
        self.n_w_1 = (84 - n_kernel_1) // stride_1 + 1
        self.n_w_2 = (self.n_w_1 - n_kernel_2) // stride_2 + 1
        self.n_kernel_1 = n_kernel_1
        self.stride_1 = stride_1
        self.n_kernel_2 = n_kernel_2
        self.stride_2 = stride_2
        self.avg_pool_1_stride = avg_pool_1_stride
        self.avg_pool_2_stride = avg_pool_2_stride
        self.avg_pool_1_k = avg_pool_1_k
        self.avg_pool_2_k = avg_pool_2_k

        self.n_avg_1 = (self.n_w_1 - avg_pool_1_k) // avg_pool_1_stride + 1
        self.n_avg_2 = (self.n_w_2 - avg_pool_2_k) // avg_pool_2_stride + 1

        if ba_config is not None:
            self.ba_filters_1_1 = ba_config['ba_filters_1_1']
            self.ba_kernel_1_1 = ba_config['ba_kernel_1_1']
            self.ba_stride_1_1 = ba_config['ba_stride_1_1']
            self.ba_filters_1_2 = ba_config['ba_filters_1_2']
            self.ba_kernel_1_2 = ba_config['ba_kernel_1_2']
            self.ba_stride_1_2 = ba_config['ba_stride_1_2']
            self.ba_filters_2 = ba_config['ba_filters_2']
            self.ba_kernel_2 = ba_config['ba_kernel_2']
            self.ba_stride_2 = ba_config['ba_stride_2']
        else:
            self.ba_filters_1_1 = 16
            self.ba_kernel_1_1 = 8
            self.ba_stride_1_1 = 4
            self.ba_filters_1_2 = 32
            self.ba_kernel_1_2 = 4
            self.ba_stride_1_2 = 2
            self.ba_filters_2 = 32
            self.ba_kernel_2 = 4
            self.ba_stride_2 = 2

        self.n_ba_1 = (self.n_w_1 - ba_config['ba_kernel_1_1']) // ba_config['ba_stride_1_1'] + 1
        self.n_ba_2 = (self.n_w_2 - ba_config['ba_kernel_2']) // ba_config['ba_stride_2'] + 1

    @property
    def output_size(self):
        if self.ba:
            return self.n_w_1 * self.n_w_1 * self.n_filters_1, \
                   self.n_w_1 * self.n_w_1 * self.n_filters_1, \
                   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
                   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
                   self.n_ba_1 * self.n_ba_1 * self.ba_filters_1_1, \
                   self.n_ba_2 * self.n_ba_2 * self.ba_filters_2
        if self.avg_ba:
            return self.n_w_1 * self.n_w_1 * self.n_filters_1, \
                   self.n_w_1 * self.n_w_1 * self.n_filters_1, \
                   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
                   self.n_w_2 * self.n_w_2 * self.n_filters_2, \
                   self.n_avg_1 * self.n_avg_1 * self.n_filters_1, \
                   self.n_avg_2 * self.n_avg_2 * self.n_filters_2
        return self.n_w_1 * self.n_w_1 * self.n_filters_1, \
               self.n_w_1 * self.n_w_1 * self.n_filters_1, \
               self.n_w_2 * self.n_w_2 * self.n_filters_2, \
               self.n_w_2 * self.n_w_2 * self.n_filters_2

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, self.n_w_1, self.n_w_1, self.n_filters_1), dtype), \
               tf.zeros((batch_size, self.n_w_2, self.n_w_2, self.n_filters_2), dtype)

    @property
    def state_size(self):
        return (self.n_w_1, self.n_w_1, self.n_filters_1), (self.n_w_2, self.n_w_2, self.n_filters_2)

    def __call__(self, inputs, state):
        v_conv_1, z_conv_1 = lif_dynamic(state[0], inputs, self.decay, self.v_th, 1.)
        if self.ba and not self.avg_ba:
            with tf.variable_scope('broadcast_1'):
                c1_r = snt.Conv2D(
                    self.ba_filters_1_1,
                    self.ba_kernel_1_1,
                    stride=self.ba_stride_1_1,
                    padding='VALID'
                )(z_conv_1)
                c1_r = snt.BatchFlatten()(c1_r)

                z_conv_1 = tf.stop_gradient(z_conv_1)
        elif self.avg_ba:
            with tf.variable_scope('broadcast_1'):
                c1_r = tf.nn.avg_pool(
                    z_conv_1, ksize=self.avg_pool_1_k, 
                    strides=self.avg_pool_1_stride, padding='VALID')
                c1_r = snt.BatchFlatten()(c1_r)
                z_conv_1 = tf.stop_gradient(z_conv_1)

        i_conv_2 = snt.Conv2D(self.n_filters_2, self.n_kernel_2, stride=self.stride_2, padding='VALID', use_bias=True)(z_conv_1)
        v_conv_2, z_conv_2 = lif_dynamic(state[1], i_conv_2, self.decay, self.v_th, 1.)
        if self.ba and not self.avg_ba:
            with tf.variable_scope('broadcast_2'):
                layer_c2_r = snt.Conv2D(
                    self.ba_filters_2,
                    self.ba_kernel_2,
                    stride=self.ba_stride_2,
                    padding='VALID'
                )
                c2_r = layer_c2_r(z_conv_2)
                c2_r = snt.BatchFlatten()(c2_r)

                z_conv_2 = tf.stop_gradient(z_conv_2)
        elif self.avg_ba:
            with tf.variable_scope('broadcast_2'):
                c2_r = tf.nn.avg_pool(
                    z_conv_2, ksize=self.avg_pool_2_k, 
                    strides=self.avg_pool_2_stride, padding='VALID')
                c2_r = snt.BatchFlatten()(c2_r)
                z_conv_2 = tf.stop_gradient(z_conv_2)
        new_state = (v_conv_1, v_conv_2)
        if self.ba or self.avg_ba:
            return (tf.reshape(z_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
                    tf.reshape(v_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
                    tf.reshape(z_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2)),
                    tf.reshape(v_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2)), c1_r, c2_r), new_state
        return (tf.reshape(z_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
                tf.reshape(v_conv_1, (-1, self.n_w_1 * self.n_w_1 * self.n_filters_1)),
                tf.reshape(z_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2)),
                tf.reshape(v_conv_2, (-1, self.n_w_2 * self.n_w_2 * self.n_filters_2))), new_state


class SpikingAgent(snt.RNNCore):
    def __init__(self, action_set, rnn_units, stop_gradient=False, n_rnn_step_factor=1,
                 tau=20, tau_readout=5, thr=.615, n_filters_1=16, n_filters_2=32,
                 n_kernel_1=8, n_kernel_2=4, stride_1=4, stride_2=2,
                 beta=.16, tau_adaptation=300, ba=False, avg_ba=False,
                 ba_config=None, fraction_adaptive=.4, n_refractory=3,
                 tau_scnn=1., thr_scnn=1., avg_pool_1_stride=4, avg_pool_1_k=8,
                 avg_pool_2_stride=2, avg_pool_2_k=4):
        super(SpikingAgent, self).__init__(name='agent')

        self._num_actions = len(action_set)
        self.ba = ba
        self.avg_ba = avg_ba
        self.rnn_units = rnn_units
        tau_readout = np.atleast_1d(tau_readout)
        self.decay = np.exp(-1 / tau_readout)
        self.n_rnn_step_factor = n_rnn_step_factor
        self.thr = thr
        self.n_filters_1 = n_filters_1
        self.n_kernel_1 = n_kernel_1
        self.stride_1 = stride_1
        self.n_filters_2 = n_filters_2
        self.n_kernel_2 = n_kernel_2
        self.stride_2 = stride_2
        self.no_linear = True
        if ba_config is not None:
            self.ba_filters_1_1 = ba_config['ba_filters_1_1']
            self.ba_kernel_1_1 = ba_config['ba_kernel_1_1']
            self.ba_stride_1_1 = ba_config['ba_stride_1_1']
            self.ba_filters_1_2 = ba_config['ba_filters_1_2']
            self.ba_kernel_1_2 = ba_config['ba_kernel_1_2']
            self.ba_stride_1_2 = ba_config['ba_stride_1_2']
            self.ba_filters_2 = ba_config['ba_filters_2']
            self.ba_kernel_2 = ba_config['ba_kernel_2']
            self.ba_stride_2 = ba_config['ba_stride_2']
        else:
            self.ba_filters_1_1 = 32
            self.ba_kernel_1_1 = 8
            self.ba_stride_1_1 = 4
            self.ba_filters_1_2 = 64
            self.ba_kernel_1_2 = 4
            self.ba_stride_1_2 = 2
            self.ba_filters_2 = 64
            self.ba_kernel_2 = 4
            self.ba_stride_2 = 2

        with self._enter_variable_scope():
            n_regular = int(rnn_units * (1. - fraction_adaptive))
            n_adaptive = rnn_units - n_regular
            beta = np.concatenate((np.zeros(n_regular), np.ones(n_adaptive))).astype(np.float32) * beta
            self.beta = beta
            n_w_1 = (84 - n_kernel_1) // stride_1 + 1
            n_w_2 = (n_w_1 - n_kernel_2) // stride_2 + 1
            n_input = n_w_2 * n_w_2 * n_filters_2
            self.core = CustomALIFWithReset(n_input, rnn_units, tau=tau, beta=beta, thr=thr,
                                            tau_adaptation=tau_adaptation, stop_gradients=stop_gradient,
                                            n_refractory=n_refractory)
            self.scnn = SpikingCNN(n_filter_1=n_filters_1, stride_1=stride_1, n_kernel_1=n_kernel_1, n_filter_2=n_filters_2, stride_2=stride_2, n_kernel_2=n_kernel_2, ba=ba, avg_ba=avg_ba, ba_config=ba_config, tau=tau_scnn, thr=thr_scnn, avg_pool_1_stride=avg_pool_1_stride, avg_pool_1_k=avg_pool_1_k, avg_pool_2_stride=avg_pool_2_stride, avg_pool_2_k=avg_pool_2_k)

    def initial_state(self, batch_size):
        return self.core.zero_state(batch_size, tf.float32), \
               (tf.zeros((batch_size, self._num_actions)), tf.zeros((batch_size, 1))), \
               self.scnn.zero_state(batch_size, tf.float32)

    def initial_eligibility_traces(self, batch_size):
        initial_eligibility_traces = [
            tf.tile(tf.zeros_like(self.core.w_in_var[None, ..., None]), (batch_size, 1, 1, 2)),
            tf.tile(tf.zeros_like(self.core.w_rec_var[None, ..., None]), (batch_size, 1, 1, 2))
        ]
        return initial_eligibility_traces

    def _head(self, core_output, head_state, torso_dict):
        def f(core_output):
            i_policy_logits = snt.Linear(self._num_actions, name='policy_logits')(core_output)
            i_baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)
            return i_policy_logits, i_baseline

        core_output = tf.concat((core_output, torso_dict['c1_r'], torso_dict['c2_r']), -1)
        policy = 0.
        baseline = 0.
        for decay in self.decay:
            i_policy_logits, i_baseline = snt.BatchApply(f)(core_output)
            policy += exp_convolve(i_policy_logits, decay, initializer=head_state[0])
            baseline += exp_convolve(i_baseline[..., None], decay, initializer=head_state[1])
        policy = policy[self.n_rnn_step_factor - 1::self.n_rnn_step_factor]
        baseline = baseline[self.n_rnn_step_factor - 1::self.n_rnn_step_factor]

        def g(policy):
            new_action = tf.multinomial(policy, num_samples=1,
                                        output_dtype=tf.int64)
            new_action = tf.squeeze(new_action, 1, name='new_action')
            return new_action

        new_action = snt.BatchApply(g)(policy)
        new_head_state = (policy[-1], baseline[-1])

        return AgentOutput(new_action, policy, baseline[..., 0]), new_head_state

    def _build(self, input_, core_state):
        action, env_output = input_
        env_outputs = environments.StepOutput(
            reward=env_output.reward[None, ...],
            info=nest.map_structure(lambda t: t[None, ...], env_output.info),
            done=to_bool(tf.cast(env_output.done, tf.int64)[None, ...]),
            observation=(tf.to_float(env_output.observation[0])[None, ...], tf.zeros(()))
        )
        actions = action[None, ...]
        outputs, core_state, custom_rnn_output, torso_outputs = self.unroll(actions, env_outputs, core_state)
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state, \
               custom_rnn_output, torso_outputs

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state, write_to_collection=False):
        _, _, done, _ = env_outputs

        env_outputs = environments.StepOutput(
            reward=env_outputs.reward,
            info=env_outputs.info,
            done=env_outputs.done,
            observation=(env_outputs.observation[0], tf.zeros(()))
        )

        head_state = core_state[1]
        n_time, n_batch = actions.get_shape()
        done = tf.cast(done, tf.float32)[..., None]
        expanded_dones = tf.reshape(
            tf.tile(done[:, None, ...],
                    (1, self.n_rnn_step_factor, 1, 1)), (n_time * self.n_rnn_step_factor, n_batch, -1))
        frame = tf.cast(env_outputs.observation[0], tf.float32) / 255.
        with tf.variable_scope('convnet'):
            i_conv1 = snt.BatchApply(snt.Conv2D(self.n_filters_1, self.n_kernel_1, stride=self.stride_1, padding='VALID', use_bias=True))(frame)
            shp = i_conv1.get_shape()
            i_conv1 = tf.reshape(
                tf.tile(i_conv1[:, None, ...],
                        (1, self.n_rnn_step_factor, 1, 1, 1, 1)), (n_time * self.n_rnn_step_factor, n_batch, *shp[2:]))
            i_conv1 = tf.transpose(i_conv1, (1, 0, 2, 3, 4))
            scnn_output, new_scnn_state = tf.nn.dynamic_rnn(self.scnn, i_conv1, initial_state=core_state[2])
            to_collection = dict()
            to_collection['lin_z'] = tf.zeros_like(scnn_output[1])
            to_collection['lin_act'] = tf.zeros_like(scnn_output[1])
            to_collection['c1_act'] = scnn_output[1]
            to_collection['c1_z'] = scnn_output[0]
            to_collection['c2_act'] = scnn_output[3]
            to_collection['c2_z'] = scnn_output[2]
            if self.ba or self.avg_ba:
                to_collection['c1_r'] = tf.transpose(scnn_output[4], (1, 0, 2))
                to_collection['c2_r'] = tf.transpose(scnn_output[5], (1, 0, 2))
            if write_to_collection:
                tf.add_to_collection('torso_output', to_collection)
            torso_outputs = scnn_output[2]
            expanded_dones = tf.transpose(expanded_dones, (1, 0, 2))
            dynamic_rnn_inputs = tf.concat((scnn_output[2], expanded_dones), -1)

        custom_rnn_output, core_state = tf.nn.dynamic_rnn(
            self.core, dynamic_rnn_inputs, initial_state=core_state[0])
        custom_rnn_output = nest.map_structure(switch_time_and_batch_dimension, custom_rnn_output)
        core_output = custom_rnn_output[0]
        core_output.set_shape((n_time * self.n_rnn_step_factor, n_batch, self.rnn_units))
        head_output, head_state = self._head(core_output, head_state, to_collection)
        core_state = (core_state, head_state, new_scnn_state)
        return head_output, core_state, custom_rnn_output, torso_outputs
