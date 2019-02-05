import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from time import time

from models import exp_convolve, LightLIF, pseudo_derivative, shift_by_one_time_step

FLAGS = tf.app.flags.FLAGS
##
tf.app.flags.DEFINE_integer('n_batch', 1, 'batch size of the testing set')

tf.app.flags.DEFINE_integer('n_out', 1, 'number of output neurons (number of target curves)')
tf.app.flags.DEFINE_integer('n_in', 5, 'number of input units.')
tf.app.flags.DEFINE_integer('n_rec', 5, 'number of recurrent units.')

tf.app.flags.DEFINE_integer('f0', 200, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')

tf.app.flags.DEFINE_integer('n_iter', 1000, 'number of iterations')
tf.app.flags.DEFINE_integer('seq_len', 2, 'Number of character steps')
tf.app.flags.DEFINE_integer('print_every', 10, 'print statistics every K iterations')
##
tf.app.flags.DEFINE_float('decay_factor', 0.5, 'reduce learning rate by constant factor')
tf.app.flags.DEFINE_integer('decay_every', 100, 'Decay learning rate every K iterations')

tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Base learning rate.')
tf.app.flags.DEFINE_float('learning_rate', 0.0003, 'Base learning rate.')
tf.app.flags.DEFINE_float('reg', 300., 'regularization coefficient')

tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', 0.615, 'threshold at which the LSNN neurons spike (in arbitrary units)')

tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots during training')
tf.app.flags.DEFINE_bool('stop_z_gradients', False, 'Stop gradient in the model dynamics to get equivalence between merge and BPTT')
tf.app.flags.DEFINE_bool('gradient_check', True, 'Make an online verification to see to match e-prop and the true gradients')

tf.app.flags.DEFINE_string('merge_or_bptt', 'merge', '')

# Experiment parameters
dt= 1 # time step in ms
input_f0 = FLAGS.f0 / 1000  # in kHz in coherence with the usage of ms for time
regularization_f0 = FLAGS.reg_rate / 1000

# Network parameters
tau_m = 30
tau_m_readout = 30
thr = FLAGS.thr

cell = LightLIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec, tau=tau_m, thr=thr, dt=dt, dampening_factor=FLAGS.dampening_factor,stop_z_gradients=FLAGS.stop_z_gradients)

# build the input pattern
frozen_poisson_noise_input = np.random.rand(FLAGS.n_batch, FLAGS.seq_len, FLAGS.n_in) < dt * input_f0
#frozen_poisson_noise_input = np.array([[[0,0,1,0,0],[0,0,0,0,0]]])
print('Input Spikes')
print(np.int_(frozen_poisson_noise_input[0]))
input_spikes = tf.constant(frozen_poisson_noise_input, dtype=tf.float32)


def sum_of_sines_target(n_sines=4, periods=[1000, 500, 333, 200], weights=None, phases=None):
    if periods is None:
        periods = [np.random.uniform(low=100, high=1000) for i in range(n_sines)]
    assert n_sines == len(periods)
    sines = []
    weights = np.random.uniform(low=0.5, high=2, size=n_sines) if weights is None else weights
    phases = np.random.uniform(low=0., high=np.pi * 2, size=n_sines) if phases is None else phases
    for i in range(n_sines):
        sine = np.sin(np.linspace(0 + phases[i], np.pi * 2 * (FLAGS.seq_len // periods[i]) + phases[i], FLAGS.seq_len))
        sines.append(sine * weights[i])

    output = sum(sines)
    #output = output - output[0]
    #scale = max(np.abs(np.min(output)),np.abs(np.max(output)))
    #output = output/scale

    return output


target_sinusoidal_outputs = [np.expand_dims(sum_of_sines_target(), axis=0) for i in range(FLAGS.n_out)]
target_sinusoidal_outputs = np.stack(target_sinusoidal_outputs, axis=2)

# Define the name of spike train for the different models
outs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
z, v = outs

with tf.name_scope('RecallLoss'):
    w_out = tf.Variable(np.random.randn(FLAGS.n_rec, FLAGS.n_out) / np.sqrt(FLAGS.n_rec), name='out_weight',
                        dtype=tf.float32)

    output_current = tf.einsum('bti,ik->btk', z, w_out)
    readout_decay = np.exp(-dt / tau_m_readout)
    output = exp_convolve(output_current, decay=readout_decay)
    output_error = output - target_sinusoidal_outputs

    loss = 0.5 * tf.reduce_sum(output_error ** 2)

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    loss_reg = 0.5 * tf.reduce_sum((av - regularization_f0) ** 2)

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    overall_loss = loss_reg * FLAGS.reg + loss

    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate = tf.assign(learning_rate, learning_rate * FLAGS.decay_factor)

    opt = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

    # Definition of the merge 1 gradients

    v_scaled = (v - thr) / thr
    post_term = pseudo_derivative(v_scaled, FLAGS.dampening_factor) #/ thr

    z_previous_time = shift_by_one_time_step(z)
    #input_spikes_previous_time = shift_by_one_time_step(input_spikes)

    pre_term_rec = exp_convolve(z_previous_time, decay=cell._decay)
    pre_term_in = exp_convolve(input_spikes, decay=cell._decay)

    eligibility_traces_rec = post_term[:, :, None, :] * pre_term_rec[:, :, :, None]
    eligibility_traces_in = post_term[:, :, None, :] * pre_term_in[:, :, :, None]

    eligibility_traces_in = exp_convolve(eligibility_traces_in, decay=readout_decay)
    eligibility_traces_rec = exp_convolve(eligibility_traces_rec, decay=readout_decay)

    #B_random = tf.constant(np.random.randn(FLAGS.n_rec, FLAGS.n_out) / np.sqrt(FLAGS.n_rec), dtype=tf.float32)
    B_random = w_out
    learning_signals = tf.einsum('btk,jk->btj', output_error, B_random)

    # The firing rate regularization lead to a local term that adds the learning signal
    average_firing_rate_error = (av - regularization_f0) / FLAGS.seq_len / FLAGS.n_batch
    learning_signals += average_firing_rate_error * FLAGS.reg

    dloss_dW_in = tf.reduce_sum(learning_signals[:, :, None, :] * eligibility_traces_in, axis=(0, 1))
    dloss_dW_rec = tf.reduce_sum(learning_signals[:, :, None, :] * eligibility_traces_rec, axis=(0, 1))
    dloss_dw_out = tf.reduce_sum(output_error[:, :, None, :] * pre_term_rec[:,:,:,None], axis=(0, 1))

    print_v = tf.print('voltage          \n',v[0])
    print_i = tf.print('input spikes     \n', input_spikes[0])
    print_z = tf.print('spikes           \n',z[0])
    print_p = tf.print('pseudo-derivative\n',post_term[0])

    with tf.control_dependencies([print_v, print_i, print_z, print_p]):

        var_list = [cell.w_in_var, cell.w_rec_var, w_out]
        true_gradients = tf.gradients(overall_loss, var_list)
        eprop_gradients = [dloss_dW_in, dloss_dW_rec, dloss_dw_out]


    if FLAGS.merge_or_bptt == 'merge':
        grads_and_vars = [(g, v) for g, v in zip(eprop_gradients, var_list)]
    elif FLAGS.merge_or_bptt == 'bptt':
        grads_and_vars = [(g, v) for g, v in zip(true_gradients, var_list)]

    train_step = opt.minimize(overall_loss)

# Real-time plotting
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
        ax.set_ylim([-1.,1.])
        ax.set_ylabel('Output')

        ax.plot(np.arange(FLAGS.seq_len), target_sinusoidal_outputs[batch,:, i], linestyle='dashed', label='Target',alpha=0.8)
        ax.plot(np.arange(FLAGS.seq_len), output[:, i], linestyle='solid', label='Target', alpha=0.8)

        ax.set_xlim([0, FLAGS.seq_len])

    i_pre = 0
    j_post = 0
    el_data_list = [results_values['eligibility_traces'][batch,:,i_pre,j_post],
                    results_values['pre_term'][batch,:,i_pre],
                    results_values['post_term'][batch,:,j_post],
                    results_values['learning_signals'][batch,:,j_post],
                    ]

    name_list = ['eligibility trace',
                 'term pre',
                 'term post',
                 'learning signal']

    for k, name, data in zip(range(len(name_list)), name_list, el_data_list):
        ax = ax_list[2 + FLAGS.n_out + k]
        ax.plot(np.arange(FLAGS.seq_len), data)
        ax.set_xlim([0,FLAGS.seq_len])
        ax.set_ylabel(name)

    ax_list[2 + FLAGS.n_out + 0].set_ylim([0,0.03])
    ax_list[2 + FLAGS.n_out + 1].set_ylim([0,0.3])
    ax_list[2 + FLAGS.n_out + 2].set_ylim([0,0.4])
    ax_list[2 + FLAGS.n_out + 3].set_ylim([-0.05,0.05])

    ax.set_xlabel('Time in ms')
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(0.1)


loss_list = []

results_tensors = {
    'loss': loss,
    'loss_reg': loss_reg,
    'overall_loss': overall_loss,

    'input_spikes': input_spikes,
    'z': z,
    'av': av,

    'pre_term': pre_term_rec,
    'post_term': post_term,
    'eligibility_traces': eligibility_traces_rec,
    'learning_signals': learning_signals,

    'output': output,

}

t_train = 0
for k_iter in range(FLAGS.n_iter):

    # train
    t0 = time()
    sess.run(train_step)
    t_train = time() - t0

    if np.mod(k_iter, FLAGS.decay_every) == 0 and k_iter > 0:
        sess.run(decay_learning_rate)
        print('Decaying learning rate to {:.3g}'.format(sess.run(learning_rate)))

    if FLAGS.gradient_check:
        if not (FLAGS.stop_z_gradients):
            print('Gradient check is disabled because the gradients of inter neuron dependencies are not blocked.')
        else:
            print('Gradient check')
            eprop_grads_np, true_grads_np = sess.run([eprop_gradients, true_gradients])
            for k_v,v in enumerate(var_list):
                eprop_grad = eprop_grads_np[k_v]
                true_grad = true_grads_np[k_v]

                is_correct = np.abs(eprop_grad - true_grad) < 1e-4


                if np.all(is_correct):
                    print('\t' + v.name + ' is correct.')
                else:
                    print('\t' + v.name + ' is wrong')
                    diff = eprop_grad - true_grad
                    print('E-prop')
                    print(np.array_str(eprop_grad[:5,:5],precision=4))
                    print('True gradients')
                    print(np.array_str(true_grad[:5,:5],precision=4))
                    print('Difference')
                    print(np.array_str(diff[:5,:5],precision=4))
                    print('Max difference')
                    print('indices',np.argmax(np.abs(diff)), ' val ', np.max(np.abs(diff)))

            raise ValueError()

    if np.mod(k_iter, FLAGS.print_every) == 0:
        t0 = time()
        results_values = sess.run(results_tensors)
        t_valid = time() - t0

        print('''Iteration {}, loss {:.3g} reg loss {:.3g}'''.format(k_iter, results_values['loss'], results_values['loss_reg']))
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
plt.show()