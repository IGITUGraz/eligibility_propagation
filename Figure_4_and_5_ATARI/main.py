# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass

import os
import datetime as dt
import multiprocessing as mp
import string
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import pickle as pkl
import yaml
import time

from spiking_agent import SpikingAgent
import rl_tools
from util import to_bool, switch_time_and_batch_dimension
import environments

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('voltage_reg_method', 'avg_time', '')
flags.DEFINE_string('restore_from', '', 'Restore model from')
flags.DEFINE_string('restore_cnn_from', '', 'Restore a CNN')
flags.DEFINE_string('level_name', 'atari/pong', '')
flags.DEFINE_string('comment', '', 'Comment to be stored as part of FLAGS')
flags.DEFINE_string('anim_format', 'mp4', '[mp4, gif]')
flags.DEFINE_string('result_dir', '/tmp/spiking_agent', 'Directory to hold the results')

flags.DEFINE_bool('apply_after_end', True, 'Apply gradients only at end of episodes')
flags.DEFINE_bool('use_eprop', True, 'Use stop gradients (eprop on rnn)')
flags.DEFINE_bool('do_plot', False, 'Interactive plotting')
flags.DEFINE_bool('save_anim', False, 'Save an animation of eval trajectories')
flags.DEFINE_bool('save_observations', False, 'Save observations')
flags.DEFINE_bool('ba_cnn', False, 'Broadcast alignment on CNN')
flags.DEFINE_bool('avg_ba_cnn', True, 'Broadcast alignment on CNN with avg pooling')
flags.DEFINE_bool('export_image_data', False, 'Save all eval observation images')
flags.DEFINE_bool('use_cubic_schedule', True, 'Use fixed cubic schedule')
flags.DEFINE_bool('resume_max_episode_len', True, 'Start from previous max episode len')

flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('parallel_episodes', 32, 'Number of actors.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')

flags.DEFINE_integer('n_rec', 400, 'Number of rnn units')
flags.DEFINE_integer('n_rnn_step_factor', 5, 'Number of intermediate steps for spiking network')
# CNN arch
flags.DEFINE_integer('n_filters_1', 16, 'Conv1 filters')
flags.DEFINE_integer('n_kernel_1', 8, 'Conv1 kernels')
flags.DEFINE_integer('stride_1', 4, 'Conv1 kernels')
flags.DEFINE_integer('n_filters_2', 8, 'Conv2 filters')
flags.DEFINE_integer('n_kernel_2', 4, 'Conv2 kernels')
flags.DEFINE_integer('stride_2', 1, 'Conv2 kernels')
# broadcast arch
flags.DEFINE_integer('ba_filters_1_1', 32, '')
flags.DEFINE_integer('ba_kernel_1_1', 8, '')
flags.DEFINE_integer('ba_stride_1_1', 4, '')
flags.DEFINE_integer('ba_filters_1_2', 64, '')
flags.DEFINE_integer('ba_kernel_1_2', 4, '')
flags.DEFINE_integer('ba_stride_1_2', 2, '')
#
flags.DEFINE_integer('ba_filters_2', 64, '')
flags.DEFINE_integer('ba_kernel_2', 4, '')
flags.DEFINE_integer('ba_stride_2', 2, '')
# avg pool arch
flags.DEFINE_integer('avg_pool_1_stride', 4, '')
flags.DEFINE_integer('avg_pool_1_k', 8, '')
flags.DEFINE_integer('avg_pool_2_stride', 2, '')
flags.DEFINE_integer('avg_pool_2_k', 4, '')
# --
flags.DEFINE_integer('eval_length', 200, '')
flags.DEFINE_integer('max_episode_len', 200, '')
flags.DEFINE_integer('increase_episode_len_every', int(7e7), '')
flags.DEFINE_integer('increase_episode_len_by', 1, '')
flags.DEFINE_integer('max_episode_len_jitter', 25, '')
flags.DEFINE_integer('n_refractory', 5, '')

flags.DEFINE_float('learning_rate_decay', 1., 'Factor to decay learning rate by')
flags.DEFINE_float('decay_every', 1e8, 'Decay learning rate every n frames')
flags.DEFINE_float('log_every', 5e6, 'Log every n frames')
flags.DEFINE_float('save_every', 1e7, 'Save every n frames')
flags.DEFINE_float('entropy_cost', 0.0025, 'Entropy cost')
flags.DEFINE_float('baseline_cost', 1., 'Baseline cost')
flags.DEFINE_float('rate_cost', 50., 'Spike regularization cost')
flags.DEFINE_float('voltage_cost_rnn', .0001, 'Voltage regularization cost LSNN')
flags.DEFINE_float('voltage_cost_cnn', .5, 'Voltage regularization cost SCNN')
flags.DEFINE_float('reg_factor_cnn', 1e8, 'Importance factor of regularization applied to SCNN')
flags.DEFINE_float('discounting', .99, 'Discounting factor (applied on action steps)')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('epsilon', 1., 'Adam epsilon')
flags.DEFINE_float('tau', 15, 'Membrane time constant for spiking agent')
flags.DEFINE_float('tau_readout', 2, 'Readout time constant for spiking agent')
flags.DEFINE_float('tau_adaptation', 70, 'Time constant of threshold adaptation')
flags.DEFINE_float('tau_scnn', 1., 'Time constant of SCNN')
flags.DEFINE_float('thr', 1., 'Threshold voltage')
flags.DEFINE_float('thr_scnn', .1, 'Threshold voltage')
flags.DEFINE_float('beta', .1, 'Threshold voltage')
flags.DEFINE_float('reward_scale', 1., 'Reward scaling')
flags.DEFINE_float('fraction_adaptive', .4, 'Fraction of adaptive neurons')


def compute_baseline_loss(advantages, mask=None):
    if mask is not None:
        return .5 * tf.reduce_sum(tf.square(advantages) * mask, 1)
    return .5 * tf.reduce_sum(tf.square(advantages), 1)


def compute_entropy_loss(logits, mask=None):
    policy = tf.nn.softmax(logits)
    log_policy = tf.nn.log_softmax(logits)
    entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
    if mask is not None:
        entropy_per_timestep = entropy_per_timestep * mask
    return -tf.reduce_sum(entropy_per_timestep, 1)


def compute_policy_gradient_loss(logits, actions, advantages, mask=None):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logits)
    advantages = tf.stop_gradient(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    if mask is not None:
        policy_gradient_loss_per_timestep = policy_gradient_loss_per_timestep * mask
    return tf.reduce_sum(policy_gradient_loss_per_timestep, 1)


def build_learner(agent, agent_state, env_outputs, agent_outputs, learning_rate,
                  gradients_buffer=None, episode_finished_mask=None, torso_variables=None, recurrent_variables=None):
    analysis_tensors = dict()

    learner_outputs, final_agent_state, custom_rnn_output, torso_outputs = agent.unroll(
        agent_outputs.action, env_outputs,
        agent_state, write_to_collection=True)
    rnn_v = custom_rnn_output[1][..., 0]
    rnn_thr = FLAGS.thr + agent.core.beta * custom_rnn_output[1][..., 1]
    rnn_pos = tf.nn.relu(rnn_v - rnn_thr)
    rnn_neg = tf.nn.relu(-rnn_v - rnn_thr)
    voltage_reg_rnn = tf.reduce_sum(tf.reduce_mean(tf.square(rnn_pos), 1))
    voltage_reg_rnn += tf.reduce_sum(tf.reduce_mean(tf.square(rnn_neg), 1))
    rnn_rate = tf.reduce_mean(custom_rnn_output[0], (0, 1))
    rnn_mean_rate = tf.reduce_mean(rnn_rate)
    analysis_tensors['rnn_rate'] = rnn_mean_rate
    rate_loss = tf.reduce_sum(tf.square(rnn_rate - .02)) * 1.
    torso_from_collection = tf.get_collection('torso_output')[-1]

    conv1_z = torso_from_collection['c1_z']
    conv2_z = torso_from_collection['c2_z']
    lin_z = torso_from_collection['lin_z']
    conv_1_rate = tf.reduce_mean(conv1_z, (0, 1))
    conv_2_rate = tf.reduce_mean(conv2_z, (0, 1))
    linear_rate = tf.reduce_mean(lin_z, (0, 1))
    mean_conv_1_rate = tf.reduce_mean(conv_1_rate)
    mean_conv_2_rate = tf.reduce_mean(conv_2_rate)
    mean_linear_rate = tf.reduce_mean(linear_rate)
    analysis_tensors['conv1_z'] = conv1_z
    analysis_tensors['conv2_z'] = conv2_z
    analysis_tensors['conv1_rate'] = mean_conv_1_rate
    analysis_tensors['conv2_rate'] = mean_conv_2_rate
    analysis_tensors['linear_rate'] = mean_linear_rate
    conv1_v = torso_from_collection['c1_act']
    conv2_v = torso_from_collection['c2_act']
    analysis_tensors['conv1_v'] = conv1_v
    analysis_tensors['conv2_v'] = conv2_v
    conv_pos = tf.nn.relu(conv1_v - FLAGS.thr_scnn)
    conv_neg = tf.nn.relu(-conv1_v - FLAGS.thr_scnn)
    if FLAGS.voltage_reg_method == 'avg_all':
        voltage_reg = tf.reduce_sum(tf.square(tf.reduce_mean(conv_pos, (0, 1))))
        voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_neg, (0, 1))))
    elif FLAGS.voltage_reg_method == 'avg_time':
        voltage_reg = tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_pos, 1)), 0))
        voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_neg, 1)), 0))
    conv_pos = tf.nn.relu(conv2_v - FLAGS.thr_scnn)
    conv_neg = tf.nn.relu(-conv2_v - FLAGS.thr_scnn)
    if FLAGS.voltage_reg_method == 'avg_all':
        voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_pos, (0, 1))))
        voltage_reg += tf.reduce_sum(tf.square(tf.reduce_mean(conv_neg, (0, 1))))
    elif FLAGS.voltage_reg_method == 'avg_time':
        voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_pos, 1)), 0))
        voltage_reg += tf.reduce_sum(tf.reduce_mean(tf.square(tf.reduce_mean(conv_neg, 1)), 0))
    reg_loss = rate_loss * FLAGS.rate_cost
    reg_loss += FLAGS.voltage_cost_rnn * voltage_reg_rnn
    reg_loss += FLAGS.voltage_cost_cnn * voltage_reg

    bootstrap_value = learner_outputs.baseline[-1]

    agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
    rewards = env_outputs.reward[1:]
    infos = nest.map_structure(lambda t: t[1:], env_outputs.info)
    done = to_bool(tf.cast(env_outputs.done, tf.int64)[1:])
    learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

    discounts = tf.to_float(~done) * FLAGS.discounting

    def scan_fun(_accumulator, _input):
        _discount, _reward = _input
        return _reward + _discount * _accumulator

    value_targets = tf.stop_gradient(tf.scan(
        scan_fun, (discounts, rewards), initializer=bootstrap_value, reverse=True))
    adv = value_targets - learner_outputs.baseline

    shifted_done = tf.concat((tf.cast(done[1:], tf.float32), tf.zeros((1, tf.shape(done)[1]))), 0)

    if episode_finished_mask is not None:
        finished_mask = (1 - tf.cumsum(shifted_done, 0)) * (1 - tf.cast(episode_finished_mask, tf.float32))[None]
    else:
        finished_mask = None

    pg_loss = compute_policy_gradient_loss(
        learner_outputs.policy_logits, agent_outputs.action,
        adv, mask=finished_mask)
    value_loss = FLAGS.baseline_cost * compute_baseline_loss(
        value_targets - learner_outputs.baseline, mask=finished_mask)
    entropy_loss = FLAGS.entropy_cost * compute_entropy_loss(
        learner_outputs.policy_logits, mask=finished_mask)
    p_actions = tf.nn.softmax(learner_outputs.policy_logits)

    loss_per_timestep = pg_loss + value_loss + entropy_loss
    total_loss = tf.reduce_sum(loss_per_timestep)
    total_loss += reg_loss

    total_loss_for_cnn = tf.reduce_sum(pg_loss + value_loss) / FLAGS.reg_factor_cnn + reg_loss

    num_env_frames = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=FLAGS.epsilon)
    cnn_optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=FLAGS.epsilon)
    normal_grad_norm = tf.zeros(())
    elig_grad_norm = tf.zeros(())
    rl_elig_grad_norm = tf.zeros(())
    recurrent_grads_and_vars = optimizer.compute_gradients(total_loss, var_list=recurrent_variables)
    if FLAGS.ba_cnn or FLAGS.avg_ba_cnn:
        torso_grads_and_vars = cnn_optimizer.compute_gradients(total_loss_for_cnn, var_list=torso_variables)
    else:
        torso_grads_and_vars = cnn_optimizer.compute_gradients(total_loss, var_list=torso_variables)
    grads_and_vars = [*recurrent_grads_and_vars, *torso_grads_and_vars]
    eligibility_variables = agent.core.variable_list

    for i, (g, v) in enumerate(grads_and_vars):
        for ve in eligibility_variables:
            if ve == v:
                normal_grad_norm += tf.reduce_sum(tf.square(g))
    if gradients_buffer is not None:
        add_ops = []
        for g, v in grads_and_vars:
            for g_holder, v_holder in gradients_buffer:
                if v == v_holder:
                    add_ops.append(tf.assign_add(g_holder, g))
        train_op = tf.group(*add_ops)
    else:
        train_op = tf.group(
            optimizer.apply_gradients(recurrent_grads_and_vars),
            cnn_optimizer.apply_gradients(torso_grads_and_vars))

    op = tf.no_op()
    with tf.control_dependencies([train_op, op]):
        num_env_frames_and_train = num_env_frames.assign_add(
            FLAGS.parallel_episodes * FLAGS.unroll_length * FLAGS.num_action_repeats)

    analysis_tensors['normal_grad_norm'] = normal_grad_norm
    analysis_tensors['elig_grad_norm'] = elig_grad_norm
    analysis_tensors['rl_elig_grad_norm'] = rl_elig_grad_norm
    analysis_tensors['pg_loss'] = pg_loss
    analysis_tensors['value_loss'] = value_loss
    analysis_tensors['entropy_loss'] = entropy_loss
    analysis_tensors['rate_loss'] = rate_loss * FLAGS.rate_cost
    analysis_tensors['voltage_loss_rnn'] = voltage_reg_rnn * FLAGS.voltage_cost_rnn
    analysis_tensors['voltage_loss_cnn'] = voltage_reg * FLAGS.voltage_cost_cnn
    analysis_tensors['action_distribution'] = p_actions
    analysis_tensors['value_target'] = value_targets

    return (done, infos, num_env_frames_and_train, analysis_tensors), optimizer, grads_and_vars


def train(action_set, level_names, log_dir):
    with tf.Graph().as_default():
        ba_config = dict(
            ba_filters_1_1=FLAGS.ba_filters_1_1,
            ba_kernel_1_1=FLAGS.ba_kernel_1_1,
            ba_stride_1_1=FLAGS.ba_stride_1_1,
            ba_filters_1_2=FLAGS.ba_filters_1_2,
            ba_kernel_1_2=FLAGS.ba_kernel_1_2,
            ba_stride_1_2=FLAGS.ba_stride_1_2,
            ba_filters_2=FLAGS.ba_filters_2,
            ba_kernel_2=FLAGS.ba_kernel_2,
            ba_stride_2=FLAGS.ba_stride_2
        )

        tau_readout = FLAGS.tau_readout
        agent = SpikingAgent(action_set, FLAGS.n_rec,
                             n_rnn_step_factor=FLAGS.n_rnn_step_factor,
                             tau=FLAGS.tau, tau_readout=tau_readout, thr=FLAGS.thr, beta=FLAGS.beta,
                             tau_adaptation=FLAGS.tau_adaptation,
                             n_filters_1=FLAGS.n_filters_1, n_kernel_1=FLAGS.n_kernel_1,
                             stride_1=FLAGS.stride_1,
                             n_filters_2=FLAGS.n_filters_2, n_kernel_2=FLAGS.n_kernel_2,
                             stride_2=FLAGS.stride_2, ba=FLAGS.ba_cnn, avg_ba=FLAGS.avg_ba_cnn,
                             stop_gradient=FLAGS.use_eprop,
                             fraction_adaptive=FLAGS.fraction_adaptive,
                             ba_config=ba_config,
                             n_refractory=FLAGS.n_refractory, tau_scnn=FLAGS.tau_scnn,
                             thr_scnn=FLAGS.thr_scnn,
                             avg_pool_1_stride=FLAGS.avg_pool_1_stride,
                             avg_pool_2_stride=FLAGS.avg_pool_2_stride,
                             avg_pool_1_k=FLAGS.avg_pool_1_k,
                             avg_pool_2_k=FLAGS.avg_pool_2_k,
                             )

        max_episode_length = tf.Variable(tf.constant(FLAGS.max_episode_len, tf.int32), trainable=False,
                                         name='max_episode_length')
        actual_max_episode_length = tf.Variable(tf.constant(FLAGS.max_episode_len, tf.int32), trainable=False,
                                                name='actual_max_episode_length')
        init_max_episode_len = tf.group(tf.assign(actual_max_episode_length, FLAGS.max_episode_len),
                                        tf.assign(max_episode_length, FLAGS.max_episode_len))
        if FLAGS.use_cubic_schedule:
            increase_episode_len_op = tf.assign(
                max_episode_length, tf.cast(
                    tf.math.pow(tf.math.pow(
                        tf.cast(max_episode_length, tf.float32), 1 / 3) + FLAGS.increase_episode_len_by, 3), tf.int32))
        else:
            increase_episode_len_op = tf.assign(max_episode_length, max_episode_length + FLAGS.increase_episode_len_by)
        if FLAGS.max_episode_len_jitter > 0:
            jitter_episode_len_op = tf.assign(
                actual_max_episode_length,
                max_episode_length +
                tf.cast(tf.random_uniform((), 0, 2 * FLAGS.max_episode_len_jitter), tf.int32) -
                FLAGS.max_episode_len_jitter)
        else:
            jitter_episode_len_op = tf.no_op()
        should_reset = tf.Variable(tf.zeros((), tf.int32), trainable=False, name='should_reset')
        envs = []
        for i in range(FLAGS.parallel_episodes):
            level_name = level_names[i % len(level_names)]
            env = environments.create_environment(level_name, FLAGS.num_action_repeats,
                                                  max_episode_length=actual_max_episode_length,
                                                  should_reset=should_reset)
            envs.append(env)
        res = rl_tools.create_local_states(agent, envs)
        persistent = res[0]
        first = res[1]
        structure, dummy_torso_output, dummy_custom_rnn_output = res[2:]
        actor_outputs, update_values = rl_tools.build_actors(agent, envs, level_name, FLAGS.unroll_length, action_set,
                                                             structure, first, dummy_torso_output,
                                                             dummy_custom_rnn_output)
        assign_ops = rl_tools.update_states(update_values, persistent)
        with tf.control_dependencies(nest.flatten(assign_ops)):
            actor_outputs = nest.map_structure(tf.identity, actor_outputs)

        eval_queue = mp.Queue()
        eval_env = environments.create_environment(level_names[0], FLAGS.num_action_repeats, queue=eval_queue,
                                                   max_episode_length=actual_max_episode_length)
        res = rl_tools.create_local_states(agent, [eval_env])
        eval_persistent = res[0]
        eval_first = res[1]
        structure, dummy_torso_output, dummy_custom_rnn_output = res[2:]
        eval_actor_output, eval_update_values, eval_rnn_h, eval_rnn_c = rl_tools.build_actors(
            agent, [eval_env], level_name, FLAGS.unroll_length, action_set, structure, eval_first,
            dummy_torso_output, dummy_custom_rnn_output, return_rnn_activity=True)

        eval_rnn_h = tf.reshape(eval_rnn_h, (
            FLAGS.unroll_length * FLAGS.n_rnn_step_factor, 1, *eval_rnn_h.get_shape()[3:]))
        eval_rnn_c = tf.reshape(eval_rnn_c, (
            FLAGS.unroll_length * FLAGS.n_rnn_step_factor, 1, *eval_rnn_c.get_shape()[3:]))
        eval_assign_ops = rl_tools.update_states(eval_update_values, eval_persistent)
        tt_agent_output, _, _, _ = agent((eval_actor_output.agent_outputs.action[:, -1],
                                          nest.map_structure(lambda a: a[:, -1], eval_actor_output.env_outputs)),
                                         eval_actor_output.agent_state)
        with tf.control_dependencies(nest.flatten(eval_assign_ops)):
            eval_actor_output = nest.map_structure(tf.identity, eval_actor_output)
        eval_bootstrap = tt_agent_output.baseline[0]
        eval_actor_output = eval_actor_output._replace(
            env_outputs=nest.map_structure(lambda a: a[0, 1:], eval_actor_output.env_outputs),
            agent_outputs=nest.map_structure(lambda a: a[0, 1:], eval_actor_output.agent_outputs),
            action_probabilities=eval_actor_output.action_probabilities[0])
        eval_rnn_h = eval_rnn_h[:, 0]
        eval_rnn_c = eval_rnn_c[:, 0]

        global_step = tf.get_variable(
            'num_environment_frames',
            initializer=tf.zeros_initializer(),
            shape=[],
            dtype=tf.int64,
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        np_learning_rate = FLAGS.learning_rate
        learning_rate = tf.get_variable('learning_rate',
                                        initializer=tf.cast(np_learning_rate, tf.float32),
                                        trainable=False)
        learning_rate_decay = FLAGS.learning_rate_decay
        decay_every = FLAGS.decay_every
        print(f'[ learning rate {np_learning_rate:.4e} ]')
        print(f'[ learning rate decay {learning_rate_decay:.4e} ]')
        learning_rate_placeholder = tf.placeholder(tf.float32, ())
        assign_new_learning_rate_op = tf.assign(learning_rate, learning_rate_placeholder)

        variables = tf.trainable_variables()
        torso_variables = []
        recurrent_variables = []
        for v in variables:
            if v.name.count('convnet') > 0 or v.name.count('agent/batch_apply/linear') > 0:
                torso_variables.append(v)
            else:
                recurrent_variables.append(v)
        print('[ torso variables ]')
        for v in torso_variables:
            print(v.name)
        print('[ recurrent variables ]')
        for v in recurrent_variables:
            print(v.name)
        vars_to_train = [a for a in recurrent_variables]
        vars_to_train += torso_variables

        gradients_buffer = None
        if FLAGS.apply_after_end:
            gradients_buffer = []
            initial_finished_mask = np.zeros((FLAGS.parallel_episodes,), dtype=np.bool)
            episode_finished_mask = tf.Variable(initial_finished_mask, trainable=False, name='episode_finished_mask')

            # accumulate gradients from 32 episodes
            with tf.variable_scope('gradients_buffer'):
                for v in vars_to_train:
                    buffer = tf.get_local_variable(v.op.name, initializer=tf.zeros_like(v), use_resource=True)
                    gradients_buffer.append((buffer, v))
        else:
            episode_finished_mask = None

        data_from_actors = actor_outputs._replace(
            env_outputs=nest.map_structure(switch_time_and_batch_dimension, actor_outputs.env_outputs),
            agent_outputs=nest.map_structure(switch_time_and_batch_dimension, actor_outputs.agent_outputs))
        output, optimizer, grads_and_vars = build_learner(agent, data_from_actors.agent_state,
                                                          data_from_actors.env_outputs,
                                                          data_from_actors.agent_outputs, learning_rate,
                                                          gradients_buffer=gradients_buffer,
                                                          episode_finished_mask=episode_finished_mask,
                                                          recurrent_variables=recurrent_variables,
                                                          torso_variables=torso_variables)
        if FLAGS.apply_after_end:
            episode_finished = tf.reduce_sum(tf.cast(data_from_actors.env_outputs.done[1:], tf.float32), 0) > 0.5
            episode_finished_assign = tf.assign(episode_finished_mask,
                                                tf.logical_or(episode_finished_mask, episode_finished))
            should_reset_clear_op = tf.assign(should_reset, tf.zeros_like(should_reset))
            with tf.control_dependencies([episode_finished_assign, should_reset_clear_op]):
                output = nest.map_structure(tf.identity, output)
            clear_episode_finished_op = tf.assign(episode_finished_mask, tf.zeros_like(episode_finished_mask))
            should_reset_assign_op = tf.assign(should_reset, tf.ones_like(should_reset))
        else:
            episode_finished_mask = tf.zeros(())

        if FLAGS.apply_after_end:
            apply_buffer = optimizer.apply_gradients(gradients_buffer)
            clear_and_apply_buffer_op = tf.group(
                apply_buffer,
                clear_episode_finished_op,
                *[tf.assign(g, tf.zeros_like(g)) for g, _ in gradients_buffer])

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=5.)
        last_decay = 0
        last_logger = 0
        last_saver = 0
        last_episode_len_increase = 0
        log_every = FLAGS.log_every
        torso_saver = tf.train.Saver(var_list=torso_variables)

        if FLAGS.do_plot:
            import matplotlib.pyplot as plt
            plt.ion()
        else:
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
        from matplotlib import animation
        import plot_tools
        traj_fig, traj_axes = plt.subplots(6, 1, figsize=(6, 6), sharex=False)
        traj_axes = traj_axes.reshape((-1,))

        debug_fig, debug_axes = plt.subplots(4, 4, figsize=(6, 6))
        if FLAGS.save_anim:
            obs_fig, obs_ax = plt.subplots(figsize=(4, 4))
            obs_ax.axis('off')

        performance_fig, performance_ax = plt.subplots(figsize=(6, 4))

        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        init_op = tf.group(global_init_op, local_init_op)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.9)
        config_proto = tf.ConfigProto(allow_soft_placement=True,
                                      gpu_options=gpu_options)

        t_zero = time.time()

        with tf.Session(config=config_proto) as session:
            initial_env_frames = 0
            if FLAGS.restore_from != '':
                try:
                    saver.restore(session, tf.train.latest_checkpoint(FLAGS.restore_from))
                except:
                    saver.restore(session, FLAGS.restore_from)
                session.run(local_init_op)
                initial_env_frames = session.run(output[2])
                last_logger = initial_env_frames - int(initial_env_frames / FLAGS.log_every)
                last_saver = initial_env_frames - int(initial_env_frames / FLAGS.save_every)
                last_decay = initial_env_frames - int(initial_env_frames / FLAGS.decay_every)
                last_episode_len_increase = initial_env_frames - int(initial_env_frames / FLAGS.increase_episode_len_by)
                print(f'[ variables restored from {FLAGS.restore_from} ]')
            else:
                session.run(init_op)
            if not FLAGS.resume_max_episode_len or FLAGS.restore_from == '':
                session.run(init_max_episode_len)

            if FLAGS.restore_cnn_from != '':
                print(f'[ restoring CNN from {FLAGS.restore_cnn_from} ]')
                torso_saver.restore(session, FLAGS.restore_cnn_from)

            summary_writer = tf.summary.FileWriterCache.get(log_dir)

            num_env_frames_v = 0
            previous_episode_finished_mask = np.zeros((FLAGS.parallel_episodes,), np.bool)

            performance_x = []
            performance_y = []
            performance_ystd = []

            perf = []
            pg_losses = []
            value_losses = []
            entropy_losses = []
            reg_losses = []
            voltage_losses_rnn = []
            voltage_losses_cnn = []
            elig_norms = []
            rl_elig_norms = []
            grad_norms = []
            reached_to_goals = []
            while num_env_frames_v < FLAGS.total_environment_frames:
                if num_env_frames_v - last_decay > decay_every:
                    last_decay += decay_every
                    np_learning_rate = np_learning_rate * learning_rate_decay
                    print(f'[ learning rate decayed by {learning_rate_decay:.4f} to {np_learning_rate:.4e} ]')
                    session.run(assign_new_learning_rate_op, {learning_rate_placeholder: np_learning_rate})
                else:
                    level_names_v, done_v, infos_v, num_env_frames_v, analysis_values, np_from_actors, np_epsiode_finished_mask = session.run(
                        (data_from_actors.level_name,) + output + (data_from_actors, episode_finished_mask))
                if FLAGS.save_observations:
                    observations = np_from_actors.env_outputs.observation[0]
                    rewards = np_from_actors.env_outputs.reward
                    dones = np_from_actors.env_outputs.done
                    data = dict(observation=observations, reward=rewards, done=dones)

                    save_path = os.path.join(log_dir, f'observations_{num_env_frames_v}.pkl')

                    with open(save_path, 'wb') as f:
                        pkl.dump(data, f)
                        print(f'[ saved observations in {save_path} ]')
                elig_norms.append(analysis_values['elig_grad_norm'])
                rl_elig_norms.append(analysis_values['rl_elig_grad_norm'])
                grad_norms.append(analysis_values['normal_grad_norm'])
                pg_losses.append(analysis_values['pg_loss'])
                reg_losses.append(analysis_values['rate_loss'])
                voltage_losses_rnn.append(analysis_values['voltage_loss_rnn'])
                voltage_losses_cnn.append(analysis_values['voltage_loss_cnn'])
                value_losses.append(analysis_values['value_loss'])
                entropy_losses.append(analysis_values['entropy_loss'])
                t1 = time.time()
                fps = (num_env_frames_v - initial_env_frames) / (t1 - t_zero)
                if FLAGS.apply_after_end:
                    pp = np.tile(previous_episode_finished_mask[None], (done_v.shape[0], 1))
                    done_v = np.logical_and(done_v, np.logical_not(pp))
                    previous_episode_finished_mask = np_epsiode_finished_mask

                if FLAGS.apply_after_end and np.all(np_epsiode_finished_mask):
                    session.run([clear_and_apply_buffer_op, should_reset_assign_op])

                if FLAGS.max_episode_len > 0:
                    session.run(jitter_episode_len_op)
                if num_env_frames_v > last_episode_len_increase + FLAGS.increase_episode_len_every and FLAGS.max_episode_len > 0:
                    session.run(increase_episode_len_op)
                    temp = session.run(max_episode_length)
                    last_episode_len_increase = num_env_frames_v
                    print(f'[ increased maximum episode length to {temp} ]')

                for episode_return, episode_step in zip(
                        infos_v.episode_return[done_v],
                        infos_v.episode_step[done_v]):
                    episode_frames = episode_step * FLAGS.num_action_repeats

                    reached_to_goals.append(episode_return > 0.)
                    perf.append(episode_return)
                    level_name = FLAGS.level_name

                    summary = tf.summary.Summary()
                    summary.value.add(tag=level_name + '/episode_return',
                                      simple_value=episode_return)
                    summary.value.add(tag=level_name + '/episode_frames',
                                      simple_value=episode_frames)
                    summary.value.add(tag=level_name + '/normal_grad_norm',
                                      simple_value=analysis_values['normal_grad_norm'])
                    summary.value.add(tag=level_name + '/elig_grad_norm',
                                      simple_value=analysis_values['elig_grad_norm'])
                    summary.value.add(tag=level_name + '/rl_elig_grad_norm',
                                      simple_value=analysis_values['rl_elig_grad_norm'])
                    summary.value.add(tag=level_name + '/pg_loss',
                                      simple_value=analysis_values['pg_loss'].mean())
                    summary.value.add(tag=level_name + '/value_loss',
                                      simple_value=analysis_values['value_loss'].mean())
                    summary.value.add(tag=level_name + '/rate_loss',
                                      simple_value=analysis_values['rate_loss'])
                    summary.value.add(tag=level_name + '/voltage_loss_rnn',
                                      simple_value=analysis_values['voltage_loss_rnn'])
                    summary.value.add(tag=level_name + '/voltage_loss_cnn',
                                      simple_value=analysis_values['voltage_loss_cnn'])
                    summary.value.add(tag=level_name + '/entropy_loss',
                                      simple_value=analysis_values['entropy_loss'].mean())
                    summary_writer.add_summary(summary, num_env_frames_v)

                if num_env_frames_v - last_saver > FLAGS.save_every:
                    model_save_path = saver.save(session, os.path.join(log_dir, 'model'),
                                                 global_step=num_env_frames_v)
                    print(f'[ saved model to {model_save_path} ]')
                    last_saver += FLAGS.save_every
                if num_env_frames_v - last_logger > log_every:
                    eval_actor_outputs = []
                    eval_rnn_zs = []
                    eval_rnn_hs = []
                    np_eval_output, np_eval_bootstrap, np_eval_rnn_h, np_eval_rnn_c = \
                        session.run([eval_actor_output, eval_bootstrap, eval_rnn_h, eval_rnn_c])
                    eval_actor_outputs.append(np_eval_output)
                    eval_rnn_zs.append(np_eval_rnn_h)
                    eval_rnn_hs.append(np_eval_rnn_c)
                    while np.sum(np_eval_output.env_outputs.done) < .5 \
                            and len(eval_rnn_zs) * FLAGS.unroll_length < FLAGS.eval_length:
                        np_eval_output, np_eval_bootstrap, np_eval_rnn_h, np_eval_rnn_c = \
                            session.run([eval_actor_output, eval_bootstrap, eval_rnn_h, eval_rnn_c])
                        eval_actor_outputs.append(np_eval_output)
                        eval_rnn_zs.append(np_eval_rnn_h)
                        eval_rnn_hs.append(np_eval_rnn_c)
                    eval_observations = []
                    if FLAGS.level_name.count('atari') > 0:
                        while not eval_queue.empty():
                            eval_observations.append(eval_queue.get_nowait())
                        eval_observations = np.stack(eval_observations, 0)
                    if FLAGS.export_image_data:
                        save_path = os.path.join(log_dir, f'img_dat_{num_env_frames_v // 1000}k.pkl')
                        with open(save_path, 'wb') as f:
                            pkl.dump(eval_observations, f)
                            print(f'[ image data saved to {save_path} ]')
                    if hasattr(agent.core, 'beta'):
                        beta = agent.core.beta
                    else:
                        beta = FLAGS.beta
                    plot_tools.update_traj_plot(traj_axes, eval_actor_outputs, np_eval_bootstrap,
                                                eval_rnn_zs, eval_rnn_hs, FLAGS.discounting,
                                                FLAGS.n_rnn_step_factor, beta=beta, thr=FLAGS.thr,
                                                level_name=FLAGS.level_name)
                    [ax.clear() for ax in debug_axes.reshape((-1,))]
                    supers = FLAGS.n_rnn_step_factor
                    n_w_1 = (84 - FLAGS.n_kernel_1) // FLAGS.stride_1 + 1
                    n_w_2 = (n_w_1 - FLAGS.n_kernel_2) // FLAGS.stride_2 + 1
                    z1 = analysis_values['conv1_z'].reshape(
                        (FLAGS.parallel_episodes, (FLAGS.unroll_length + 1) * supers, n_w_1, n_w_1, FLAGS.n_filters_1))
                    z2 = analysis_values['conv2_z'].reshape(
                        (FLAGS.parallel_episodes, (FLAGS.unroll_length + 1) * supers, n_w_2, n_w_2, FLAGS.n_filters_2))
                    v1 = analysis_values['conv1_v'].reshape(
                        (FLAGS.parallel_episodes, (FLAGS.unroll_length + 1) * supers, n_w_1, n_w_1, FLAGS.n_filters_1))
                    v2 = analysis_values['conv2_v'].reshape(
                        (FLAGS.parallel_episodes, (FLAGS.unroll_length + 1) * supers, n_w_2, n_w_2, FLAGS.n_filters_2))
                    obs = np_from_actors.env_outputs.observation[0][0, 0, ..., 0]
                    for ax in debug_axes.reshape((-1,)):
                        ax.spines['top'].set_visible(True)
                        ax.spines['bottom'].set_visible(True)
                        ax.spines['left'].set_visible(True)
                        ax.spines['right'].set_visible(True)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    debug_axes[0][0].imshow(obs, cmap='Greys')
                    debug_axes[1][0].axis('off')
                    for i in range(1, 4):
                        debug_axes[0][i].imshow(v1[0][0, :, :, i], cmap='Greys')
                    for i in range(1, 4):
                        debug_axes[1][i].imshow(z1[0][0, :, :, i], cmap='Greys')
                    for i in range(4):
                        debug_axes[2][i].imshow(v2[0][0, :, :, i], cmap='Greys')
                    for i in range(4):
                        debug_axes[3][i].imshow(z2[0][0, :, :, i], cmap='Greys')
                    debug_fig_path = os.path.join(log_dir, f'debug_{num_env_frames_v // 1000}k.png')
                    debug_fig.savefig(debug_fig_path, dpi=300)
                    if FLAGS.save_anim:
                        if FLAGS.level_name.count('atari') > 0:
                            obs = eval_observations
                        else:
                            obs = np.concatenate([a.env_outputs.observation[0] for a in eval_actor_outputs], 0)
                        ims = []
                        obs_ax.clear()
                        obs_ax.axis('off')
                        for i in range(obs.shape[0]):
                            if len(obs[i].shape) == 3 and obs[i].shape[-1] == 1:
                                im = obs_ax.imshow(obs[i][..., 0], cmap='Greys')
                            else:
                                im = obs_ax.imshow(obs[i])
                            ims.append([im])
                        ani = animation.ArtistAnimation(obs_fig, ims, interval=50, blit=True, repeat_delay=500)
                        if FLAGS.anim_format == 'mp4':
                            anim_fig_path = os.path.join(log_dir, f'anim_{num_env_frames_v // 1000}k.mp4')
                            ani.save(anim_fig_path, fps=30)
                        else:
                            if FLAGS.anim_format != 'gif':
                                print('[ unknown anim_format, assuming gif ]')
                            anim_fig_path = os.path.join(log_dir, f'anim_{num_env_frames_v // 1000}k.gif')
                            ani.save(anim_fig_path, writer='imagemagick', fps=30)
                    traj_fig_path = os.path.join(log_dir, f'value_{num_env_frames_v // 1000}k.pdf')
                    traj_fig.savefig(traj_fig_path)
                    if FLAGS.do_plot:
                        plt.draw()
                        plt.pause(.1)

                    last_logger += log_every
                    date_str = dt.datetime.now().strftime('%H:%M:%S %d-%m-%y')
                    print(f'[ {num_env_frames_v / 1e6:.3f} M frames at {date_str} ]')
                    if len(perf) > 0:
                        m_perf = np.mean(perf)
                        s_perf = np.std(perf)
                        performance_x.append(num_env_frames_v)
                        performance_y.append(m_perf)
                        performance_ystd.append(s_perf)
                        _p = dict(x=performance_x, y=performance_y, ystd=performance_ystd)
                        plot_tools.update_performance_plot(performance_ax, _p)
                        fig_path = os.path.join(log_dir, f'performance.pdf')
                        performance_fig.savefig(fig_path)
                        print(f'  - performance mean {m_perf:.2f}')
                        print(f'                 std {s_perf:.2f}')
                        print(f'  - episodes with positive reward {np.mean(reached_to_goals) * 100:.1f} %')
                    else:
                        print('  ! no episodes completed')
                    print(f'  - fps {fps:.0f}')
                    rnn_rate = analysis_values['rnn_rate']
                    print(f'  - rnn rate {rnn_rate:.3f}')
                    conv1_rate = analysis_values['conv1_rate']
                    conv2_rate = analysis_values['conv2_rate']
                    linear_rate = analysis_values['linear_rate']
                    print(f'  - conv1 rate {conv1_rate:.3f}'
                          f'  - conv2 rate {conv2_rate:.3f}'
                          f'  - linear rate {linear_rate:.3f}')

                    print(f'  - policy loss {np.mean(pg_losses):.4f}'
                          f'  - value loss {np.mean(value_losses):.4f}'
                          f'  - entropy loss {np.mean(entropy_losses):.4f}'
                          f'  - rate loss {np.mean(reg_losses):.4f}'
                          f'  - voltage loss rnn {np.mean(voltage_losses_rnn):.4f}'
                          f'  - voltage loss cnn {np.mean(voltage_losses_cnn):.4f}')
                    perf = []
                    pg_losses = []
                    value_losses = []
                    entropy_losses = []
                    reg_losses = []
                    voltage_losses_rnn = []
                    voltage_losses_cnn = []
                    grad_norms = []
                    elig_norms = []
                    rl_elig_norms = []
                    reached_to_goals = []


def main(_):
    level_names = [FLAGS.level_name]
    action_set = list(range(10))

    if FLAGS.level_name.count('pong') > 0:
        action_set = [0, 3, 4]

    root_dir = os.path.expanduser(FLAGS.result_dir)
    identifier = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(4))
    sim_name = 'rl_{}'.format(identifier)

    experiment_path = os.path.join(root_dir, sim_name)
    os.makedirs(experiment_path, exist_ok=True)
    print('[ simulation name: {} ]'.format(sim_name))

    print('[ results path: {} ]'.format(experiment_path))

    with open(os.path.join(experiment_path, 'config.yaml'), 'w') as f:
        yaml.dump(FLAGS.flag_values_dict(), f, allow_unicode=True, default_flow_style=False)

    train(action_set, level_names, experiment_path)


if __name__ == '__main__':
    tf.app.run()
