from time import time
import matplotlib.pyplot as plt
from matplotlib import collections as mc, patches
import numpy as np
import numpy.random as rd
import tensorflow as tf

from models import LightALIF, exp_convolve
from storerecall_utils import generate_storerecall_data, error_rate, gen_custom_delay_batch

FLAGS = tf.app.flags.FLAGS

##
tf.app.flags.DEFINE_integer('n_batch', 128, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('n_charac', 2, 'number of characters in the recall task')
tf.app.flags.DEFINE_integer('n_in', 100, 'number of input units.')
tf.app.flags.DEFINE_integer('n_rec', 100, 'number of recurrent units.')

tf.app.flags.DEFINE_integer('f0', 50, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 300, 'number of iterations')
tf.app.flags.DEFINE_integer('seq_len', 12, 'Number of character steps')
tf.app.flags.DEFINE_integer('seq_delay', 6, 'Expected delay in character steps. Must be <= seq_len - 2')
tf.app.flags.DEFINE_integer('tau_char', 200, 'Duration of symbols')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('print_every', 1, 'Decay every')
##
tf.app.flags.DEFINE_float('beta', 1.7, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 1200, 'Mikolov model alpha - threshold decay')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in LSNN and output neurons')
tf.app.flags.DEFINE_float('learning_rate', 0.002, 'Base learning rate.')
tf.app.flags.DEFINE_float('reg', 1, 'regularization coefficient')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, '')
tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', .03, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('fraction_adaptive', .4, 'Fraction of adaptive neurons')
##
tf.app.flags.DEFINE_bool('broadcast_alignment', False, 'Use broadcast aligment to train network')
tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')

tf.app.flags.DEFINE_bool('stop_z_gradients', True,
                         'stop gradients in the model dynamics to get mathematical equivalence between eprop and BPTT')

# Run asserts to check seq_delay and seq_len relation is ok
_ = gen_custom_delay_batch(FLAGS.seq_len, FLAGS.seq_delay, 1)

# Fix the random seed if given as an argument
if FLAGS.seed >= 0:
    seed = FLAGS.seed
else:
    seed = rd.randint(10 ** 6)
rd.seed(seed)
tf.set_random_seed(seed)

# Experiment parameters
dt = 1.
print_every = FLAGS.print_every

# Frequencies
input_f0 = FLAGS.f0 / 1000  # in kHz in coherence with the usgae of ms for time
regularization_f0 = FLAGS.reg_rate / 1000

# Network parameters
tau_v = FLAGS.tau_out
thr = FLAGS.thr

decay = np.exp(-dt / FLAGS.tau_out)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
# Symbol number
n_charac = FLAGS.n_charac  # Number of digit symbols
n_input_symbols = n_charac + 2  # Total number of symbols including recall and store
n_output_symbols = n_charac  # Number of output symbols
recall_symbol = n_input_symbols - 1  # ID of the recall symbol
store_symbol = n_input_symbols - 2  # ID of the store symbol

# Neuron population sizes
input_neuron_split = np.array_split(np.arange(FLAGS.n_in), n_input_symbols)

# Generate the cell
n_adaptive = int(FLAGS.n_rec * FLAGS.fraction_adaptive)
n_regular = FLAGS.n_rec - n_adaptive
beta = np.concatenate([np.zeros(n_regular), np.ones(n_adaptive) * FLAGS.beta])
cell = LightALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_rec, tau=tau_v, beta=beta, thr=thr,
                 dt=dt, tau_adaptation=FLAGS.tau_a, dampening_factor=FLAGS.dampening_factor,
                 stop_z_gradients=FLAGS.stop_z_gradients)
zero_state = cell.zero_state(FLAGS.n_batch, tf.float32)

# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
input_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                            name='InputNums')  # Lists of input character for the recall task
target_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                             name='TargetNums')  # Lists of target characters of the recall task
recall_mask = tf.placeholder(dtype=tf.bool, shape=(None, None),
                             name='RecallMask')  # Binary tensor that points to the time of presentation of a recall

# Other placeholder that are useful for computing accuracy and debuggin
target_sequence = tf.placeholder(dtype=tf.int64, shape=(None, None),
                                 name='TargetSequence')  # The target characters with time expansion
recall_charac_mask = tf.equal(input_nums, recall_symbol, name='RecallCharacMask')


def get_data_dict(batch_size, seq_len=FLAGS.seq_len, override_input=None):
    p_sr = 1/(1 + FLAGS.seq_delay)
    spk_data, is_recall_data, target_seq_data, memory_seq_data, in_data, target_data = generate_storerecall_data(
        batch_size=batch_size,
        f0=input_f0,
        sentence_length=seq_len,
        n_character=FLAGS.n_charac,
        n_charac_duration=FLAGS.tau_char,
        n_neuron=FLAGS.n_in,
        prob_signals=p_sr,
        with_prob=True,
        override_input=override_input,
    )
    data_dict = {input_spikes: spk_data, input_nums: in_data, target_nums: target_data, recall_mask: is_recall_data,
                 target_sequence: target_seq_data}

    return data_dict

# Define the name of spike train for the different models
outputs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=zero_state)
z, v, thr = outputs


# random feedback weights
@tf.custom_gradient
def BA_logits(psp, W_out, B_out):
    logits = tf.einsum('btj,jk->btk', psp, W_out)

    def grad(dy):
        dloss_dw_out = tf.einsum('bij,bik->jk', psp, dy)
        dloss_db_out = tf.zeros_like(B_out)
        dloss_dpsp = tf.einsum('bik,jk->bij', dy, B_out)
        return [dloss_dpsp, dloss_dw_out, dloss_db_out]
    return logits, grad


with tf.name_scope('RecallLoss'):
    tiled_target_nums = tf.reshape(tf.tile(target_nums[..., None], (1, 1, FLAGS.tau_char)),
                                   (FLAGS.n_batch, FLAGS.seq_len * FLAGS.tau_char))
    tiled_recall_charac_mask = tf.reshape(tf.tile(recall_charac_mask[..., None], (1, 1, FLAGS.tau_char)),
                                          (FLAGS.n_batch, FLAGS.seq_len * FLAGS.tau_char))
    # target_nums_at_recall = tf.boolean_mask(target_nums, recall_charac_mask)
    target_nums_at_recall = tf.boolean_mask(tiled_target_nums, tiled_recall_charac_mask)
    Y = tf.one_hot(target_nums_at_recall, depth=n_output_symbols, name='Target')

    # MTP models do not use controller (modulator) population for output
    out_neurons = z
    n_neurons = out_neurons.get_shape()[2]
    psp = exp_convolve(out_neurons, decay=decay)

    w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])
    b_out = tf.constant(rd.randn(FLAGS.n_rec, n_output_symbols)
                        / np.sqrt(FLAGS.n_rec), dtype=tf.float32)

    if FLAGS.broadcast_alignment:
        out = BA_logits(psp, w_out, b_out)
    else:
        out = tf.einsum('btj,jk->btk', psp, w_out)

    Y_predict = tf.boolean_mask(out, tiled_recall_charac_mask, name='Prediction')

    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_nums_at_recall,
                                                                                logits=Y_predict))

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)

    _, recall_errors, false_sentence_id_list = error_rate(out, target_nums, input_nums, n_charac)

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    regularization_coeff = tf.Variable(np.ones(n_neurons) * FLAGS.reg, dtype=tf.float32, trainable=False)

    loss_reg = tf.reduce_sum(tf.square(av - regularization_f0) * regularization_coeff)

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    # scaling loss_recall to match order of magnitude of loss from script_recall.py
    # this is needed to keep the same regularization coefficients (reg, regl2) across scripts
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)

    loss = loss_reg + loss_recall

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = opt.minimize(loss=loss, global_step=global_step)

# Real-time plotting
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Open an interactive matplotlib window to plot in real time
if FLAGS.do_plot:
    plt.ion()
    fig, ax_list = plt.subplots(4, figsize=(5.9, 6))


def update_plot(plot_result_values, batch=0, n_max_neuron_per_raster=20, n_max_synapses=n_adaptive):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    for k_data, data, d_name in zip(range(2),
                                    [plot_result_values['input_spikes'], plot_result_values['z']],
                                    ['Input', 'Hidden']):

        ax = ax_list[k_data]
        ax.grid(color='black', alpha=0.15, linewidth=0.4)

        if np.size(data) > 0:
            data = data[batch]
            n_max = min(data.shape[1], n_max_neuron_per_raster)
            cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
            data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
            ax.imshow(data.T, aspect='auto', cmap='hot_r')
            ax.set_ylabel(d_name)
            ax.set_xticklabels([])

            if d_name == 'Input':
                ax.set_yticklabels([])
                n_channel = data.shape[1] // n_input_symbols
                ax.add_patch(  # Value 0 row
                    patches.Rectangle((0, 0), data.shape[0], n_channel, facecolor="red", alpha=0.15))
                ax.add_patch(  # Value 1 row
                    patches.Rectangle((0, n_channel), data.shape[0], n_channel, facecolor="blue", alpha=0.15))
                ax.add_patch(  # Store row
                    patches.Rectangle((0, 2 * n_channel), data.shape[0], n_channel, facecolor="yellow", alpha=0.15))
                ax.add_patch(  # Recall row
                    patches.Rectangle((0, 3 * n_channel), data.shape[0], n_channel, facecolor="green", alpha=0.15))

                top_margin = 0.08
                left_margin = 0.01
                ax.text(left_margin, 1. - top_margin, 'Recall', transform=ax.transAxes,
                        fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.75 - top_margin, 'Store', transform=ax.transAxes,
                        fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.5 - top_margin, 'Value 1', transform=ax.transAxes,
                        fontsize=7, verticalalignment='top')
                ax.text(left_margin, 0.25 - top_margin, 'Value 0', transform=ax.transAxes,
                        fontsize=7, verticalalignment='top')

    # plot targets
    ax = ax_list[2]
    mask = plot_result_values['recall_charac_mask'][batch]
    data = plot_result_values['target_nums'][batch]
    data[np.invert(mask)] = -1
    lines = []
    ind_nt = np.argwhere(data != -1)
    for idx in ind_nt.tolist():
        i = idx[0]
        lines.append([(i * FLAGS.tau_char, data[i]), ((i + 1) * FLAGS.tau_char, data[i])])
    lc_t = mc.LineCollection(lines, colors='green', linewidths=2, label='Target')
    ax.add_collection(lc_t)  # plot target segments

    # plot softmax of psp-s per dt for more intuitive monitoring
    # ploting only for second class since this is more intuitive to follow (first class is just a mirror)
    output2 = plot_result_values['out_plot'][batch, :, 1]
    presentation_steps = np.arange(output2.shape[0])
    ax.set_yticks([0, 0.5, 1])
    ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Output')
    line_output2, = ax.plot(presentation_steps, output2, color='purple', label='Output', alpha=0.7)
    ax.axis([0, presentation_steps[-1] + 1, -0.3, 1.1])
    ax.legend(handles=[lc_t, line_output2], loc='lower center', fontsize=7,
              bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.set_xticklabels([])

    # debug plot for psp-s or biases
    ax.set_xticklabels([])
    ax = ax_list[-1]
    ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Threshold')
    sub_data = plot_result_values['thr'][batch]
    vars = np.var(sub_data, axis=0)
    # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses * 3:3]
    cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses]
    presentation_steps = np.arange(sub_data.shape[0])
    ax.plot(sub_data[:, cell_with_max_var], color='r', label='Output', alpha=0.4, linewidth=1)
    ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
             np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]

    ax.set_xlabel('Time in ms')
    plt.draw()
    plt.pause(1)

test_loss_list = []
test_loss_with_reg_list = []
validation_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []
results_tensors = {
    'loss': loss,
    'loss_reg': loss_reg,
    'loss_recall': loss_recall,
    'recall_errors': recall_errors,
    'final_state': final_state,
    'av': av,
    'regularization_coeff': regularization_coeff,
}

results_tensors['w_in_val'] = cell.w_in_val
results_tensors['w_rec_val'] = cell.w_rec_val
results_tensors['w_out'] = w_out

w_in_last = sess.run(cell.w_in_val)
w_rec_last = sess.run(cell.w_rec_val)
w_out_last = sess.run(w_out)

plot_result_tensors = {'input_spikes': input_spikes,
                       'z': z,
                       'thr': thr,
                       'input_nums': input_nums,
                       'target_nums': target_nums,
                       }

t_train = 0
t_ref = time()
for k_iter in range(FLAGS.n_iter):
    # Monitor the training with a validation set
    t0 = time()
    val_dict = get_data_dict(FLAGS.n_batch)

    plot_result_tensors['psp'] = psp
    plot_result_tensors['out_plot'] = out_plot
    plot_result_tensors['recall_charac_mask'] = recall_charac_mask
    plot_result_tensors['Y'] = Y
    plot_result_tensors['Y_predict'] = Y_predict
    plot_result_tensors['thr'] = thr

    results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)
    t_run = time() - t0

    # Storage of the results
    test_loss_with_reg_list.append(results_values['loss_reg'])
    test_loss_list.append(results_values['loss_recall'])
    validation_error_list.append(results_values['recall_errors'])
    training_time_list.append(t_train)
    time_to_ref_list.append(time() - t_ref)

    if np.mod(k_iter, print_every) == 0:

        print('''Iteration {}, statistics on the validation set average error {:.2g} +- {:.2g} (trial averaged)'''
              .format(k_iter, np.mean(validation_error_list[-print_every:]),
                      np.std(validation_error_list[-print_every:])))

        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max

        firing_rate_stats = get_stats(results_values['av'] * 1000)
        reg_coeff_stats = get_stats(results_values['regularization_coeff'])

        if FLAGS.do_plot:
            update_plot(plot_results_values)

        print('''
        firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t
        average {:.0f} +- std {:.0f} (averaged over batches and time)
        reg. coeff        min {:.2g} \t max {:.2g} \t average {:.2g} +- std {:.2g}

        comput. time (s)  training {:.2g} \t validation {:.2g}
        loss              classif. {:.2g} \t reg. loss  {:.2g}
        '''.format(
            firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
            firing_rate_stats[2], firing_rate_stats[3],
            reg_coeff_stats[0], reg_coeff_stats[1], reg_coeff_stats[2], reg_coeff_stats[3],
            t_train, t_run,
            results_values['loss_recall'], results_values['loss_reg']
        ))

    # train
    train_dict = get_data_dict(FLAGS.n_batch)
    t0 = time()
    final_state_value, _ = sess.run([final_state, train_step], feed_dict=train_dict)
    t_train = time() - t0
