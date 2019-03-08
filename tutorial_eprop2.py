import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
import datetime
import os

from models_eprop2 import LIF, spike_encode, exp_convolve, pseudo_derivative
from robot_trajectories import RobotTrajectories

tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('n_rec_error_module', 300, 'Number of hidden units in error module')
tf.app.flags.DEFINE_integer('n_rec_network', 400, 'number of recurrent units.')
tf.app.flags.DEFINE_integer('n_in', 10, 'number of input neurons.')

tf.app.flags.DEFINE_integer('n_ref', 5, 'refractory period')
tf.app.flags.DEFINE_integer('reg_rate', 20, 'target rate for regularization')
tf.app.flags.DEFINE_integer('error_module_reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 20000, 'number of iterations')
tf.app.flags.DEFINE_integer('seq_len', 1000, 'Number of character steps')
tf.app.flags.DEFINE_integer('n_train_trials', 1, 'Number of trials')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')

tf.app.flags.DEFINE_float('tau', 20, 'time constant')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Base learning rate.')
tf.app.flags.DEFINE_float('reg', 100, 'regularization coefficient')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, '')
tf.app.flags.DEFINE_float('thr', .4, 'threshold at which the RSNN neurons spike')
tf.app.flags.DEFINE_float('inner_learning_rate', 1e-4, 'inner loop learning rate')

tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('interactive_plot', False, 'Make interactive plots')

FLAGS = tf.app.flags.FLAGS

# Fix the random seed if given as an argument
if FLAGS.seed >= 0:
    seed = FLAGS.seed
else:
    seed = rd.randint(10 ** 6)
rd.seed(seed)
tf.set_random_seed(seed)

dt = 1.
discrete_decay = np.exp(-dt / FLAGS.tau)  # discrete decay with time constant tau
n_out = 2  # number of output neurons (two motor commands)
trial_length = 500


def raster_plot(ax, spikes, linewidth=0.8, time_offset=0, **kwargs):

    n_t, n_n = spikes.shape
    event_times,event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike] + time_offset
    event_ids = event_ids[:max_spike]

    for n,t in zip(event_ids,event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)

    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([time_offset, n_t + time_offset])
    ax.set_yticks([0, n_n])


# define model
with tf.variable_scope('network'):
    cell = LIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec_network, tau=FLAGS.tau, thr=FLAGS.thr, n_refractory=FLAGS.n_ref)

    # equip model weights with a batch dimension
    w_in = tf.tile(cell.w_in_val[None, ...], (FLAGS.batch_size, 1, 1))
    w_rec = tf.tile(cell.w_rec_val[None, ...], (FLAGS.batch_size, 1, 1))
    cell.set_weights(w_in, w_rec)

    # output weights
    w_out_init = 0.05 * rd.randn(FLAGS.n_rec_network, 2) / np.sqrt(FLAGS.n_rec_network)
    w_out_init = tf.cast(w_out_init, tf.float32)
    w_out = tf.get_variable('w_out', initializer=w_out_init)

# define error module
with tf.variable_scope('error_module'):
    n_error_module_in = n_out * 100  # population encoded
    n_error_module_in += FLAGS.n_rec_network + FLAGS.n_in

    error_module = LIF(n_error_module_in, FLAGS.n_rec_error_module, thr=FLAGS.thr, n_refractory=FLAGS.n_ref)
    error_module_zero_state = error_module.zero_state(FLAGS.batch_size, tf.float32)

    error_w_out_init = rd.randn(FLAGS.n_rec_error_module, FLAGS.n_rec_network) / np.sqrt(FLAGS.n_rec_error_module)
    error_w_out_init = tf.cast(error_w_out_init, tf.float32)
    learning_signal_projection = tf.get_variable('learning_signal_projection', initializer=error_w_out_init)


sine_family = RobotTrajectories(FLAGS.batch_size, FLAGS.seq_len, FLAGS.n_train_trials + 1, dt_step=dt / 1000)
data_set = tf.data.Dataset.from_generator(sine_family, (tf.float32, tf.float32, tf.float32, tf.float32),
                                          sine_family.shape).prefetch(30)

# generate input spikes, clock like signal
input_spikes = np.zeros((int(FLAGS.seq_len / (FLAGS.n_train_trials + 1)), FLAGS.n_in), np.float32)
n_of_steps = 5
input_spike_every = 10
assert FLAGS.seq_len % n_of_steps == 0
step_len = int(FLAGS.seq_len / (FLAGS.n_train_trials + 1) / n_of_steps)
step_group = int(FLAGS.n_in / n_of_steps)
for i in range(n_of_steps):
    input_spikes[i*step_len:(i+1)*step_len:input_spike_every, i*step_group:(i+1)*step_group] = 1.
trial_input = input_spikes
input_spikes = np.concatenate([input_spikes] * (FLAGS.n_train_trials + 1), 0)
input_spikes = tf.tile(tf.constant(input_spikes)[None, ...], (FLAGS.batch_size, 1, 1))
trial_input = tf.tile(tf.constant(trial_input)[None, ...], (FLAGS.batch_size, 1, 1))

it = data_set.make_one_shot_iterator()
target_x, target_y, t_omega0, t_omega1 = it.get_next()
target_x.set_shape(sine_family.shape[0])
target_y.set_shape(sine_family.shape[1])
all_targets = tf.stack((target_x, target_y), -1)
omega = tf.stack((t_omega0, t_omega1), -1)

# initial configuration of the arm
np_initial = np.array([0., np.pi / 2], np.float32)
init_position = tf.tile(tf.constant(np_initial)[None, :], (FLAGS.batch_size, 1))

cartesian_loss = 0.
cell_zero_state = cell.zero_state(FLAGS.batch_size, tf.float32)
inner_loop_variables = [w_in, w_rec]
psp_in = exp_convolve(trial_input, discrete_decay)

inner_loop_output = dict()
inner_loop_output['learning_signals'] = tf.zeros((FLAGS.batch_size, 0, FLAGS.n_rec_network))
inner_loop_output['inner_loop_loss'] = tf.zeros((FLAGS.batch_size, 0))
inner_loop_output['motor_commands'] = tf.zeros((FLAGS.batch_size, 0, 2))
inner_loop_output['error_spikes'] = tf.zeros((FLAGS.batch_size, 0, FLAGS.n_rec_error_module))
inner_loop_output['z'] = tf.zeros((FLAGS.batch_size, 0, FLAGS.n_rec_network))
inner_loop_output['cartesian'] = tf.zeros((FLAGS.batch_size, 0, 2))
inner_loop_output['psp'] = tf.zeros((FLAGS.batch_size, 0, FLAGS.n_rec_network))

for i in range(FLAGS.n_train_trials + 1):
    with tf.variable_scope('inner_loop', reuse=tf.AUTO_REUSE):
        # simulate main network
        cell_output, _ = tf.nn.dynamic_rnn(cell, trial_input, initial_state=cell_zero_state)
        z, v = cell_output
        z_pre = tf.concat((cell_zero_state.z[:, None, :], z), 1)
        filtered = exp_convolve(z_pre, discrete_decay)
        e_rec = filtered[:, :-1]
        psp = filtered[:, 1:]

        motor_commands = tf.einsum('bti,ij->btj', psp, w_out)
        arm_configuration = init_position[:, None, :] + tf.cumsum(motor_commands, axis=1) * dt / 1000

        phi0 = arm_configuration[..., 0]
        phi1 = arm_configuration[..., 1] + phi0

        cartesian_x = (tf.cos(phi0) + tf.cos(phi1)) * .5
        cartesian_y = (tf.sin(phi0) + tf.sin(phi1)) * .5
        cartesian = tf.stack((cartesian_x, cartesian_y), -1)

        cartesian_error = cartesian - all_targets[:, :trial_length, :]
        cartesian_loss = tf.reduce_mean(tf.square(cartesian_error), -1)

        # compose error inputs
        error_inputs = all_targets[:, :trial_length, :]
        clipped_error = tf.clip_by_value(error_inputs, -1., 1.)
        encoded_error_input = spike_encode(clipped_error, -1., 1.)
        assert len(encoded_error_input.get_shape()) == 4
        error_inputs = tf.reshape(encoded_error_input, (FLAGS.batch_size, trial_length, -1))
        error_inputs = tf.concat((error_inputs, z, trial_input), -1)

        # simulate the error module
        error_output, _ = tf.nn.dynamic_rnn(error_module, error_inputs, initial_state=error_module_zero_state)
        error_spikes, error_v = error_output
        error_filtered_spikes = exp_convolve(error_spikes, discrete_decay)

        # compute learning signal projection
        learning_signals = tf.einsum('bti,ij->btj', error_filtered_spikes, learning_signal_projection)

        # compute and apply weight update if training trial
        if i < FLAGS.n_train_trials:
            v_scaled = (v - FLAGS.thr) / FLAGS.thr
            p = pseudo_derivative(v_scaled, cell.dampening_factor)

            # learning signal (j) x post factor (j) x eligibility trace (ji)
            dw_rec = tf.einsum('btj,bti->bij', learning_signals * p, e_rec)
            dw_rec = tf.where(tf.tile(tf.constant(cell.recurrent_disconnect_mask[None, ...]), (FLAGS.batch_size, 1, 1)),
                              tf.zeros_like(dw_rec), dw_rec)
            inner_loop_updates = [tf.einsum('btj,bti->bij', learning_signals * p, psp_in),
                                  dw_rec]
            inner_loop_variables = [a - FLAGS.inner_learning_rate * b for a, b in zip(inner_loop_variables,
                                                                                      inner_loop_updates)]
            cell.set_weights(*inner_loop_variables)

        # store tensors for use in outer loop optimization
        inner_loop_output['learning_signals'] = tf.concat((inner_loop_output['learning_signals'], learning_signals), 1)
        inner_loop_output['inner_loop_loss'] = tf.concat((inner_loop_output['inner_loop_loss'], cartesian_loss), 1)
        inner_loop_output['motor_commands'] = tf.concat((inner_loop_output['motor_commands'], motor_commands), 1)
        inner_loop_output['error_spikes'] = tf.concat((inner_loop_output['error_spikes'], error_spikes), 1)
        inner_loop_output['z'] = tf.concat((inner_loop_output['z'], z), 1)
        inner_loop_output['cartesian'] = tf.concat((inner_loop_output['cartesian'], cartesian), 1)
        inner_loop_output['psp'] = tf.concat((inner_loop_output['psp'], psp), 1)

# outer loop optimization
with tf.name_scope('outer_loop_loss'):
    cartesian_loss = tf.reduce_mean(inner_loop_output['inner_loop_loss'][:, -trial_length:])
    motor_loss = tf.reduce_mean(tf.square(inner_loop_output['motor_commands'] - omega)[:, -trial_length:, :])
    outer_loop_loss = 0.5 * (motor_loss * .1 + cartesian_loss)

# neuron spike rate regularization
with tf.name_scope('regularization'):
    target_rate = FLAGS.reg_rate / 1000
    error_module_target_rate = FLAGS.error_module_reg_rate / 1000

    error_module_av = tf.reduce_mean(inner_loop_output['error_spikes'], axis=(0, 1))
    error_module_loss_reg = tf.reduce_sum(tf.square(error_module_av - error_module_target_rate))

    av = tf.reduce_mean(inner_loop_output['z'], axis=(0, 1))
    network_loss_reg = tf.reduce_sum(tf.square(av - target_rate))

    loss_reg = FLAGS.reg * (network_loss_reg + error_module_loss_reg)

# Aggregate the losses
with tf.name_scope('optimization'):
    learning_rate = tf.get_variable('learning_rate', initializer=tf.cast(FLAGS.learning_rate, dtype=tf.float32),
                                    trainable=False)
    loss = loss_reg + outer_loop_loss
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = opt.minimize(loss)
print('graph complete')

tightened = False

all_params = tf.trainable_variables()
print('parameters')
for p in all_params:
    print(p.name)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Real-time plotting
# Open an interactive matplotlib window to plot in real time
if FLAGS.do_plot and FLAGS.interactive_plot:
    plt.ion()
if FLAGS.do_plot:
    fig, ax_list = plt.subplots(8, FLAGS.n_train_trials + 1, figsize=(5.7, 8))
    fig.canvas.set_window_title('e-prop 2')


def update_plot(tightened, plot_result_values, batch_ind=0):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    ylabel_x = -0.15
    ylabel_y = 0.5
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        for l in range(ax_list.shape[1]):
            ax = ax_list[k][l]
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.clear()

    for ind_demo in range(FLAGS.n_train_trials + 1):
        t_slice = np.arange(trial_length * ind_demo, trial_length * (ind_demo + 1))

        # PLOT Input signals
        ax = ax_list[0][ind_demo]
        data = plot_result_values['input_spikes']
        data = data[batch_ind][t_slice]
        raster_plot(ax, data, linewidth=0.4, color='black')
        if ind_demo == 0:
            ax.set_ylabel('input')
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        else:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        ax.xaxis.set_ticks([0, trial_length - 1])
        ax.set_xticklabels([])

        # PLOT OUTPUT AND TARGET
        ax = ax_list[1][ind_demo]
        data = plot_result_values['Y'][batch_ind]
        for l in range(1):
            line_target, = ax.plot(t_slice, data[t_slice, l], '--', color='blue', label='target', alpha=0.7)
        output2 = plot_result_values['position'][batch_ind]
        ax.set_yticks([-1, 0, 1])
        if ind_demo == 0:
            ax.set_ylabel('position $x$')
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        else:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])
        for l in range(1):
            line_output2, = ax.plot(t_slice, output2[t_slice, l], color='blue', label='network', alpha=0.7)
        ax.set_xlim([t_slice[0], t_slice[-1] + 1])
        ax.set_ylim([-1.2, 1.2])
        if ind_demo == 0:
            ax.legend(handles=[line_output2, line_target], loc='lower left', fontsize=7, ncol=3)
        ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])
        ax.set_xticklabels([])

        # PLOT OUTPUT AND TARGET
        ax = ax_list[2][ind_demo]
        data = plot_result_values['Y'][batch_ind]
        for l in range(1, 2):
            line_target, = ax.plot(t_slice, data[t_slice, l], '--', color='blue', label='target', alpha=0.7)
        output2 = plot_result_values['position'][batch_ind]
        ax.set_yticks([-1, 0, 1])
        if ind_demo == 0:
            ax.set_ylabel('position $y$')
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        else:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])
        for l in range(1, 2):
            line_output2, = ax.plot(t_slice, output2[t_slice, l], color='blue', label='network', alpha=0.7)
        ax.set_xlim([t_slice[0], t_slice[-1] + 1])
        ax.set_ylim([-1.2, 1.2])
        if ind_demo == 0:
            ax.legend(handles=[line_output2, line_target], loc='lower left', fontsize=7, ncol=3)
        ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])
        ax.set_xticklabels([])

        ax = ax_list[3][ind_demo]
        output2 = plot_result_values['motor_commands'][batch_ind]
        target_omega = plot_result_values['omega'][batch_ind]
        ax.set_yticks([-20, 0, 20])
        if ind_demo == 0:
            ax.set_ylabel('motor $\dot \phi_1$')
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        else:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])

        # MOTOR COMMANDS
        motor_color = 'r'
        line_output2, = ax.plot(t_slice, output2[t_slice, 0], color=motor_color, label='network', alpha=0.7)
        line_target, = ax.plot(t_slice, target_omega[t_slice, 0], '--', color=motor_color, alpha=0.6, label='target')
        ax.set_xlim([t_slice[0], t_slice[-1] + 1])
        ax.set_ylim(np.array([-1.2, 1.2]) * 20)
        if ind_demo == 0:
            ax.legend(loc='lower left', fontsize=7, ncol=3)
        ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])
        ax.set_xticklabels([])

        ax = ax_list[4][ind_demo]
        output2 = plot_result_values['motor_commands'][batch_ind]
        target_omega = plot_result_values['omega'][batch_ind]
        ax.set_yticks([-20, 0, 20])
        # ax.grid(color='black', alpha=0.15, linewidth=0.4)
        if ind_demo == 0:
            ax.set_ylabel('motor $\dot \phi_2$')
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        else:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])
        # MOTOR COMMANDS
        line_output2, = ax.plot(t_slice, output2[t_slice, 1], color=motor_color, label='network', alpha=0.7)
        line_target, = ax.plot(t_slice, target_omega[t_slice, 1], '--', color=motor_color, alpha=0.6, label='target')
        ax.set_xlim([t_slice[0], t_slice[-1] + 1])
        ax.set_ylim(np.array([-1.2, 1.2]) * 20)
        if ind_demo == 0:
            ax.legend(loc='lower left', fontsize=7, ncol=3)
        ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])
        ax.set_xticklabels([])

        # PLOT SPIKES
        ax = ax_list[5][ind_demo]
        data = plot_result_values['z']
        data = data[batch_ind]
        raster_plot(ax, data[t_slice, :40], time_offset=ind_demo * trial_length, linewidth=0.3)
        if ind_demo == 0:
            ax.set_ylabel('network')
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        else:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])
        ax.set_xticklabels([])

        ax = ax_list[6][ind_demo]
        data = plot_result_values['error_spikes']
        data = data[batch_ind]
        raster_plot(ax, data[t_slice, :40], time_offset=ind_demo * trial_length, linewidth=0.3)
        if ind_demo == 0:
            ax.set_ylabel('error module')
            ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
        else:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_ticks([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])

        ax = ax_list[-1][ind_demo]
        if ind_demo < FLAGS.n_train_trials:
            ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])
            ax.set_xticklabels([])
            if ind_demo == 0:
                ax.set_ylabel('$\\widehat L_j$')
                ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
            else:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks([])
            sub_data = plot_result_values['learning_signals'][batch_ind]
            vars = np.var(sub_data, axis=0)
            cell_with_max_var = np.argsort(vars)[::-1][:FLAGS.n_rec_network]
            time_sliced_data = sub_data[t_slice]
            ax.plot(t_slice, time_sliced_data[:, cell_with_max_var], color='k', label='output', alpha=0.4, linewidth=1)
            ax.axis([t_slice[0], t_slice[-1], np.min(sub_data[:, cell_with_max_var]),
                     np.max(sub_data[:, cell_with_max_var])])
            ax.xaxis.set_ticks([t_slice[0], t_slice[-1]])
            ax.xaxis.set_ticklabels([t_slice[0], t_slice[-1] + 1])
            if ind_demo == 0:
                ax.set_xlabel('time in ms')
        else:
            ax.axis('off')

    # To plot with interactive python one need to wait one second to the time to draw the axis
    if not tightened:
        fig.tight_layout()
    if FLAGS.interactive_plot:
        plt.draw()
        plt.pause(1)
    return fig


results_tensors = dict()
results_tensors['loss'] = loss
results_tensors['loss_reg'] = loss_reg
results_tensors['loss_outer_loop'] = outer_loop_loss
results_tensors['av'] = av
results_tensors['error_module_av'] = error_module_av
results_tensors['inner_loop_loss'] = inner_loop_output['inner_loop_loss']
results_tensors['w_in_val'] = cell.w_in_val
results_tensors['w_rec_val'] = cell.w_rec_val
results_tensors['w_out'] = w_out

plot_result_tensors = dict()
plot_result_tensors['input_spikes'] = input_spikes
plot_result_tensors['z'] = inner_loop_output['z']
plot_result_tensors['position'] = inner_loop_output['cartesian']
plot_result_tensors['learning_signals'] = inner_loop_output['learning_signals']
plot_result_tensors['error_spikes'] = inner_loop_output['error_spikes']
plot_result_tensors['Y'] = all_targets
plot_result_tensors['omega'] = omega
plot_result_tensors['psp'] = inner_loop_output['psp']
plot_result_tensors['motor_commands'] = inner_loop_output['motor_commands']

print_every = 20

for k_iter in range(FLAGS.n_iter):
    if k_iter % print_every == 0:
        results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors])

        print('|> Iteration {} - {}'.format(k_iter, datetime.datetime.now().strftime('%H:%M:%S %b %d, %Y')))
        print('|  -- outer loop loss {:5.3f}'.format(results_values['loss_outer_loop']))
        print('|  -- regularization loss {:5.3f}'.format(results_values['loss_reg']))
        print('|  -- average firing rate {:5.3f}, {:5.3f} (error module)'.format(
            np.mean(results_values['av']), np.mean(results_values['error_module_av'])))
        print('|' + '_' * 100)

        if FLAGS.do_plot:
            f1 = update_plot(tightened, plot_results_values)
            f1.savefig(os.path.expanduser('~/temp_fig.png'), dpi=300)
        tightened = True

    sess.run(train_step)
