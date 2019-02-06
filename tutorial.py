import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from time import time

from models import exp_convolve, LightLIF, pseudo_derivative, shift_by_one_time_step, check_gradients, \
    sum_of_sines_target

FLAGS = tf.app.flags.FLAGS
##
tf.app.flags.DEFINE_integer('n_batch', 1, 'batch size of the testing set')

tf.app.flags.DEFINE_integer('n_out', 1, 'number of output neurons (number of target curves)')
tf.app.flags.DEFINE_integer('n_in', 100, 'number of input units')
tf.app.flags.DEFINE_integer('n_rec', 100, 'number of recurrent units')

tf.app.flags.DEFINE_integer('f0', 50, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')

tf.app.flags.DEFINE_integer('n_iter', 1000, 'number of iterations')
tf.app.flags.DEFINE_integer('seq_len', 1000, 'number of time steps per sequence')
tf.app.flags.DEFINE_integer('print_every', 10, 'print statistics every K iterations')

tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'dampening factor to stabilize learning in RNNs')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
tf.app.flags.DEFINE_float('reg', 300., 'regularization coefficient')

tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', 0.03, 'threshold at which the LSNN neurons spike (in arbitrary units)')

tf.app.flags.DEFINE_bool('truncate_eligibility_trace', False, 'truncate the eligibility traces to simplify the SpiNNaker implementation')
tf.app.flags.DEFINE_bool('do_plot', True, 'interactive plots during training')
tf.app.flags.DEFINE_bool('random_feedback', False,
                         'use random feedback if true, otherwise take the symmetric of the readout weights')
tf.app.flags.DEFINE_bool('stop_z_gradients', True,
                         'stop gradients in the model dynamics to get mathematical equivalence between eprop and BPTT')
tf.app.flags.DEFINE_bool('gradient_check', True,
                         'verify that the gradients computed with e-prop match the gradients of BPTT')

tf.app.flags.DEFINE_string('eprop_or_bptt', 'eprop', 'choose the learing rule, it should be `eprop` of `bptt`')

# Experiment parameters
dt = 1  # time step in ms
input_f0 = FLAGS.f0 / 1000  # input firing rate in kHz in coherence with the usage of ms for time
regularization_f0 = FLAGS.reg_rate / 1000  # desired average firing rate in kHz
tau_m = tau_m_readout = 30
thr = FLAGS.thr

cell = LightLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec, tau=tau_m, thr=thr, dt=dt,
                dampening_factor=FLAGS.dampening_factor,
                stop_z_gradients=FLAGS.stop_z_gradients)

# build the input pattern
frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
input_spikes = tf.constant(frozen_poisson_noise_input, dtype=tf.float32)

# build the output pattern (note that the [None,:] adds an extra dimension of size 1 to the tensor)
target_sinusoidal_outputs = [sum_of_sines_target(FLAGS.seq_len)[None, :] for i in range(FLAGS.n_out)]
target_sinusoidal_outputs = np.stack(target_sinusoidal_outputs, axis=2)

# Tensorflow ops that simulates the RNN
# (tensorflow ops are symbolic objects, it will only be computed with sees.run(..) is called)
outs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
z, v = outs

with tf.name_scope('RecallLoss'):
    w_out = tf.Variable(np.random.randn(FLAGS.n_rec, FLAGS.n_out) / np.sqrt(FLAGS.n_rec),
                        name='out_weight',
                        dtype=tf.float32)

    # Tensorflow ops defining the readout neuron
    output_current = tf.einsum('bti,ik->btk', z, w_out)
    readout_decay = np.exp(-dt / tau_m_readout)
    output = exp_convolve(output_current, decay=readout_decay)

    # Tensorflow op defining the loss function
    output_error = output - target_sinusoidal_outputs
    loss = 0.5 * tf.reduce_sum(output_error ** 2)

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Tensorflow op of the loss for the firing rate regularization
    # We ask the plasticity to reduce the mean square between the average firing of each neuron and a target firing rate
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    average_firing_rate_error = av - regularization_f0
    loss_reg = 0.5 * tf.reduce_sum(average_firing_rate_error ** 2)

    overall_loss = loss_reg * FLAGS.reg + loss

# Aggregate the losses
with tf.name_scope('E-prop'):
    # To compute e-prop explicitly we defined below the eligibility traces and the learning signals
    # Some useful values
    v_scaled = (v - thr) / thr # voltage scaled to be 0 at threshold and -1 at rest
    post_term = pseudo_derivative(v_scaled, FLAGS.dampening_factor) / thr # non-linear function of the voltage
    z_previous_time = shift_by_one_time_step(z) # z(t-1) instead of z(t)

    pre_term_w_in = exp_convolve(input_spikes, decay=cell._decay) \
        if not FLAGS.truncate_eligibility_trace else input_spikes
    pre_term_w_rec = exp_convolve(z_previous_time, decay=cell._decay) \
        if not FLAGS.truncate_eligibility_trace else z_previous_time
    pre_term_w_out = exp_convolve(z, decay=cell._decay)

    eligibility_traces_w_in = post_term[:, :, None, :] * pre_term_w_in[:, :, :, None]
    eligibility_traces_w_rec = post_term[:, :, None, :] * pre_term_w_rec[:, :, :, None]

    # To define the gradient of the readout error,
    # the eligibility traces are smoothed with the same filter as the readout
    eligibility_traces_convolved_w_in = exp_convolve(eligibility_traces_w_in, decay=readout_decay)
    eligibility_traces_convolved_w_rec = exp_convolve(eligibility_traces_w_rec, decay=readout_decay)

    # To define the gradient of the regularization error defined on the averaged firing rate,
    # the eligibility traces should be averaged over time
    eligibility_traces_averaged_w_in = tf.reduce_mean(eligibility_traces_w_in, axis=(0, 1))
    eligibility_traces_averaged_w_rec = tf.reduce_mean(eligibility_traces_w_rec, axis=(0, 1))

    if FLAGS.random_feedback:
        B_random = tf.constant(np.random.randn(FLAGS.n_rec, FLAGS.n_out) / np.sqrt(FLAGS.n_rec), dtype=tf.float32)
    else:
        B_random = w_out # better performance is obtained with the true error feed-backs
    learning_signals = tf.einsum('btk,jk->btj', output_error, B_random)

    # gradients of the main loss with respect to the weights
    dloss_dw_out = tf.reduce_sum(output_error[:, :, None, :] * pre_term_w_out[:, :, :, None], axis=(0, 1))
    dloss_dw_in = tf.reduce_sum(learning_signals[:, :, None, :] * eligibility_traces_convolved_w_in, axis=(0, 1))
    dloss_dw_rec = tf.reduce_sum(learning_signals[:, :, None, :] * eligibility_traces_convolved_w_rec, axis=(0, 1))

    # gradients of the regularization loss with respect to the weights
    dreg_loss_dw_in = average_firing_rate_error * eligibility_traces_averaged_w_in
    dreg_loss_dw_rec = average_firing_rate_error * eligibility_traces_averaged_w_rec

    # combine the gradients
    dloss_dw_in += dreg_loss_dw_in * FLAGS.reg
    dloss_dw_rec += dreg_loss_dw_rec * FLAGS.reg

    # Somewhat important detail: self connection are disabled therefore gradients on the diagonal are zeros
    mask_autotapse = np.diag(np.ones(FLAGS.n_rec, dtype=bool))
    dloss_dw_rec = tf.where(mask_autotapse, tf.zeros_like(dloss_dw_rec), dloss_dw_rec)

    # put the resulting gradients into lists
    var_list = [cell.w_in_var, cell.w_rec_var, w_out]
    true_gradients = tf.gradients(overall_loss, var_list)
    eprop_gradients = [dloss_dw_in, dloss_dw_rec, dloss_dw_out]

with tf.name_scope("Optimization"):
    opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    if FLAGS.eprop_or_bptt == 'eprop':
        grads_and_vars = [(g, v) for g, v in zip(eprop_gradients, var_list)]
    elif FLAGS.eprop_or_bptt == 'bptt':
        grads_and_vars = [(g, v) for g, v in zip(true_gradients, var_list)]

    # Each time we run this tensorflow operation a weight update is applied
    train_step = opt.minimize(overall_loss)

# Initialize the tensorflow session (until now we only built a computational graph, no simulaiton has been performed)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Open an interactive matplotlib window to plot in real time
if FLAGS.do_plot:
    plt.ion()
    fig, ax_list = plt.subplots(2 + FLAGS.n_out + 4, figsize=(8, 12), sharex=True)

def update_plot(plot_result_values, batch=0, n_max_neuron_per_raster=40):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """

    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()

    # PLOT Input signals
    for k, spike_ref in enumerate(['input_spikes', 'z']):
        spikes = plot_result_values[spike_ref][batch]

        spikes = spikes[:, :n_max_neuron_per_raster]
        ax = ax_list[k]

        ax.imshow(spikes.T, aspect='auto', cmap='hot_r')
        ax.set_xlim([0, FLAGS.seq_len])
        ax.set_ylabel(spike_ref)

    output = plot_result_values['output'][batch]
    for i in range(output.shape[1]):
        ax = ax_list[i + 2]
        ax.set_yticks([-1, 0, 1])
        # ax.set_ylim([-1., 1.])
        ax.set_ylabel('Output')

        ax.plot(np.arange(FLAGS.seq_len), target_sinusoidal_outputs[batch, :, i], linestyle='dashed', label='Target',
                alpha=0.8)
        ax.plot(np.arange(FLAGS.seq_len), output[:, i], linestyle='solid', label='Target', alpha=0.8)

        ax.set_xlim([0, FLAGS.seq_len])

    el_data_list = []
    i_pre = j_post = 0
    while i_pre == j_post or np.all(el_data_list[2] == 0):
        # choose i and j to find an interesting synapse representative of what is happening
        i_pre = np.random.randint(FLAGS.n_rec)
        j_post = np.random.randint(FLAGS.n_rec)
        el_data_list = [results_values['pre_term'][batch, :, i_pre],
                        results_values['post_term'][batch, :, j_post],
                        results_values['eligibility_traces'][batch, :, i_pre, j_post],
                        results_values['learning_signals'][batch, :, j_post],
                        ]

    name_list = ['term pre',
                 'term post',
                 'eligibility trace',
                 'learning signal']

    for k, name, data in zip(range(len(name_list)), name_list, el_data_list):
        ax = ax_list[2 + FLAGS.n_out + k]
        ax.plot(np.arange(FLAGS.seq_len), data)
        ax.set_xlim([0, FLAGS.seq_len])
        ax.set_ylabel(name)

    ax_list[2 + FLAGS.n_out + 0].set_ylim([0, 0.25])
    ax_list[2 + FLAGS.n_out + 1].set_ylim([0, 10.])
    ax_list[2 + FLAGS.n_out + 2].set_ylim([0, 1.])
    ax_list[2 + FLAGS.n_out + 3].set_ylim([-0.08, 0.08])

    ax.set_xlabel('Time in ms')
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(0.1)


# Loss list to store the loss over itertaions
loss_list = []

# dictionary of tensors that we want to compute simultaneously (most of them are just computed for plotting)
results_tensors = {
    'loss': loss,
    'loss_reg': loss_reg,
    'overall_loss': overall_loss,

    'input_spikes': input_spikes,
    'z': z,
    'av': av,

    'pre_term': pre_term_w_rec,
    'post_term': post_term,
    'eligibility_traces': eligibility_traces_convolved_w_rec,
    'learning_signals': learning_signals,

    'output': output,

}

t_train = 0
for k_iter in range(FLAGS.n_iter):

    # train
    t0 = time()
    sess.run(train_step)
    t_train = time() - t0

    if np.mod(k_iter, FLAGS.print_every) == 0:

        if FLAGS.gradient_check:
            if not (FLAGS.stop_z_gradients):
                print('Gradient check is disabled because the gradients of inter neuron dependencies are not blocked.')
            elif FLAGS.seq_len > 20:
                print('Gradient check is disabled for sequence lengths larger than 20 time steps')
            else:
                print('Gradient check')
                eprop_grads_np, true_grads_np = sess.run([eprop_gradients, true_gradients])
                check_gradients(var_list, eprop_grads_np, true_grads_np)

        # Run the simulation
        t0 = time()
        results_values = sess.run(results_tensors)
        t_valid = time() - t0

        print('''Iteration {}, loss {:.3g} reg loss {:.3g}'''.format(k_iter, results_values['loss'],
                                                                     results_values['loss_reg']))
        loss_list.append(results_values['loss'])


        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max


        firing_rate_stats = get_stats(results_values['av'] * 1000)

        print('''
            firing rate (Hz)  min {:.0f} \t max {:.0f} \t average {:.0f} +- std {:.0f} (averaged over batches and time)
            comput. time (s)  training {:.2g} \t validation {:.2g}'''.format(
            firing_rate_stats[0], firing_rate_stats[1], firing_rate_stats[2], firing_rate_stats[3],
            t_train, t_valid,
        ))

        if FLAGS.do_plot:
            update_plot(results_values)

plt.ioff()
update_plot(results_values)

fig, ax_res = plt.subplots()
ax_res.plot(loss_list)
ax_res.set_xlabel('iterations')
ax_res.set_ylabel('mean square error')

plt.show()
