import datetime
import os
import socket
from time import time
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from evidence_accumulation_tools import update_plot, generate_click_task_data, save_file
from models import EligALIF, exp_convolve

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'

FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()
# training parameters
tf.app.flags.DEFINE_integer('n_batch', 64, 'batch size')
tf.app.flags.DEFINE_integer('n_iter', 2000, 'total number of iterations')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'Base learning rate.')
tf.app.flags.DEFINE_float('stop_crit', 0.07, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('print_every', 10, 'Print every')
tf.app.flags.DEFINE_integer('validate_every', 10, 'validate every')

# training algorithm
tf.app.flags.DEFINE_bool('eprop', False, 'Use e-prop to train network')
tf.app.flags.DEFINE_string('eprop_impl', 'autodiff', '["autodiff", "hardcoded"] -> use tensorflow for computing e-prop '
                                                     'updates or implement equations directly')
tf.app.flags.DEFINE_string('feedback', 'symmetric', '["random", "symmetric"] Use random or symmetric e-prop')

# neuron model and simulation parameters
tf.app.flags.DEFINE_float('beta', 1.7, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 2000, 'Mikolov model alpha - threshold decay [ms]')
tf.app.flags.DEFINE_float('thr', 0.6, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('tau_v', 20, 'tau for PSP decay in LSNN  neurons [ms]')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in output neurons [ms]')
tf.app.flags.DEFINE_float('reg_f', 1, 'regularization coefficient for firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target firing rate for regularization [Hz]')
tf.app.flags.DEFINE_integer('n_ref', 5, 'Number of refractory steps [ms]')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'factor that controls amplitude of pseudoderivative')
tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')

# other settings
tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('device_placement', False, '')

# Fix the random seed if given as an argument
if FLAGS.seed >= 0:
    seed = FLAGS.seed
else:
    seed = rd.randint(10 ** 6)

assert FLAGS.eprop_impl in ['autodiff', 'hardcoded']
assert FLAGS.feedback in ['random', 'symmetric']

rd.seed(seed)
tf.set_random_seed(seed)

# Experiment parameters
dt = FLAGS.dt
print_every = FLAGS.print_every
t_delay = np.random.uniform(1050, 1050)
t_cue_spacing = 150
n_cues = 7
t_recall = 150

# Symbol numbers
n_input_symbols = 4  # Total number of symbols including random noise and recall
n_output_symbols = 2
recall_symbol = n_input_symbols - 1  # ID of the recall symbol
store_symbol = n_input_symbols - 2  # ID of the store symbol

# Frequencies
input_f0 = 40. / 1000.  # in kHz in coherence with the usgae of ms for time
regularization_f0 = FLAGS.reg_rate / 1000.

# Network parameters
tau_v = FLAGS.tau_v
thr = FLAGS.thr
n_adaptive = 50
n_regular = 50
n_neurons = n_adaptive + n_regular
decay = np.exp(-dt / FLAGS.tau_out)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
n_in = 40

# Neuron population sizes
input_neuron_split = np.array_split(np.arange(n_in), n_input_symbols)

# Generate the cell
tau_a_list = FLAGS.tau_a * np.ones(n_adaptive)

rhos = np.exp(-1 / tau_a_list)
beta_a = FLAGS.beta * (1 - rhos) / (1 - np.exp(-1 / FLAGS.tau_v))
beta = np.concatenate([np.zeros(n_regular), beta_a])
# generate threshold decay time constants
tau_a = np.concatenate([np.ones(n_regular), tau_a_list])


def get_data_dict(batch_size):
    seq_len = int(t_cue_spacing * n_cues + t_delay + t_recall)
    spk_data, in_nums, target_data, _ = generate_click_task_data(
        batch_size=batch_size,
        seq_len=seq_len,
        n_neuron=n_in,
        recall_duration=t_recall,
        p_group=0.3,
        t_cue=100,
        n_cues=n_cues,
        t_interval=t_cue_spacing,
        f0=input_f0,
        n_input_symbols=n_input_symbols)

    data_dict = {input_spikes: spk_data, input_nums: in_nums, target_nums: target_data}

    return data_dict


# generate folder name for results
cell_name = "ALIF"
if FLAGS.eprop:
    training_algorithm = FLAGS.feedback + "eprop"
else:
    training_algorithm = "bptt"
print('\n -------------- \n' + cell_name + '\n -------------- \n')
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_reference = '{}_{}_in{}_R{}_A{}_lr{}_{}'.format(
    time_stamp, cell_name, n_in, n_regular, n_adaptive, FLAGS.learning_rate,
    training_algorithm)
print('FILE REFERENCE: ' + file_reference)

# Generate input placeholders
input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
input_nums = tf.placeholder(dtype=tf.float32, shape=(None, None),
                            name='InputSpikes')  # MAIN input spike placeholder

target_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                             name='TargetNums')  # Lists of target characters of the recall task


# build computational graph
with tf.variable_scope('CellDefinition'):
    cell = EligALIF(n_in=n_in, n_rec=n_regular + n_adaptive, tau=tau_v, beta=beta, thr=thr,
                    dt=dt, tau_adaptation=tau_a, dampening_factor=FLAGS.dampening_factor,
                    stop_z_gradients=FLAGS.eprop, n_refractory=FLAGS.n_ref)
    zero_state = cell.zero_state(FLAGS.n_batch, tf.float32)

with tf.name_scope('RNNOutputs'):
    outputs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=zero_state)
    z, s = outputs
    v, b = s[..., 0], s[..., 1]

with tf.name_scope('RecallLoss'):
    w_out = tf.get_variable(name='out_weight', shape=[n_regular + n_adaptive, n_output_symbols])
    w_out_var = w_out
    if FLAGS.feedback == 'random':
        # generate random feedback matrix
        b_out_vals = rd.randn(n_regular + n_adaptive, n_output_symbols)
        b_out = tf.constant(b_out_vals, dtype=tf.float32, name='broadcast_weights')

    @tf.custom_gradient
    def matmul_random_feedback(psp, w_out_arg, b_out_arg):
        # use this function to generate the random feedback path
        logits = tf.einsum('btj,jk->btk', psp, w_out_arg)

        def grad(dy):
            dloss_dw_out = tf.einsum('bij,bik->jk', psp, dy)
            dloss_dpsp = tf.einsum('bik,jk->bij', dy, b_out_arg)
            dloss_db_out = tf.zeros_like(b_out_arg)
            return [dloss_dpsp, dloss_dw_out, dloss_db_out]

        return logits, grad

    out_neurons = z
    psp = exp_convolve(out_neurons, decay=decay)

    if FLAGS.eprop and FLAGS.feedback == 'random':
        out = matmul_random_feedback(psp, w_out, b_out)
    else:
        out = tf.einsum('btj,jk->btk', psp, w_out)

    # we only use network output at the end for classification
    output_logits = out[:, -t_cue_spacing:]
    tiled_targets = tf.tile(target_nums[:, np.newaxis, -1], (1, t_cue_spacing))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tiled_targets,
                                                                         logits=output_logits))
    y_predict = tf.argmax(tf.reduce_mean(output_logits, axis=1), axis=1)

    # Define the accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(target_nums[:, -1], y_predict), dtype=tf.float32))
    recall_errors = 1 - accuracy

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out[:2], axis=-1)

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization+
    ad_thr = FLAGS.thr + b * beta
    v_scaled = (v - ad_thr) / FLAGS.thr
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    regularization_coeff = tf.Variable(np.ones(n_neurons) * FLAGS.reg_f, dtype=tf.float32, trainable=False)
    loss_reg_f = tf.reduce_sum(tf.square(av - regularization_f0) * regularization_coeff)

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    loss = loss_reg_f + loss
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    var_list = [cell.w_in_var, cell.w_rec_var, w_out_var]

    if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':
        # use recursive computation of e-traces in cell
        learning_signal = tf.gradients(loss, z)[0]
        # e-traces for input synapses
        grad_in, e_trace, _, epsilon_a = cell.compute_loss_gradient(learning_signal, input_spikes, z, v, b,
                                                                    zero_on_diagonal=False)
        # e-traces for recurrent synapses
        rec_spikes = tf.concat([tf.zeros_like(z[:, 0])[:, None], z[:, :-1]], axis=1)
        grad_rec, _, _, _ = cell.compute_loss_gradient(learning_signal, rec_spikes, z, v, b, zero_on_diagonal=True)
        # gradients for output weights
        grad_out = tf.gradients(loss, w_out_var)[0]
        # concatenate all gradients
        gradient_list = [grad_in, grad_rec, grad_out]
        # for comparision with autodiff version
        true_gradient_list = tf.gradients(loss, var_list)
        # check that tensorflows autodiff produces same result
        g_name = ['in', 'rec', 'out']
        grad_checks = [tf.debugging.Assert(tf.reduce_all(tf.abs(g1 - g2) < 1e-4), data=[nn, g1, g2]) for g1, g2, nn in
                       zip(gradient_list, true_gradient_list, g_name)]
    else:
        # This automatically computes the correct gradients in tensor flow
        learning_signal = tf.zeros_like(z)
        grad_checks = []
        gradient_list = tf.gradients(loss, var_list)

    grads_and_vars = [(g, v) for g, v in zip(gradient_list, var_list)]

    with tf.control_dependencies(grad_checks):
        train_step = opt.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

# create session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.device_placement))
sess.run(tf.global_variables_initializer())

# variables for storing results and plots
if FLAGS.do_plot:
    plt.ion()

    if FLAGS.eprop and FLAGS.eprop_impl == 'hardcoded':
        # plot learning signal and traces - only works in hardcoded mode!
        n_subplots = 7 - int(n_regular == 0) - int(n_adaptive == 0)
    else:
        n_subplots = 4
    fig, ax_list = plt.subplots(n_subplots, figsize=(5.9, 6))
    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.set_window_title(socket.gethostname() + ' - ' + training_algorithm)

validation_loss_list = []
validation_error_list = []
training_time_list = []
n_iter_list = []

results_tensors = {
    'loss_recall': loss,
    'loss_reg': loss_reg_f,
    'recall_errors': recall_errors,
    'av': av,
    'regularization_coeff': regularization_coeff,
}

plot_result_tensors = {'input_spikes': input_spikes,
                       'input_nums': input_nums,
                       'z': z,
                       'z_grad': learning_signal,
                       'thr': tf.constant(thr),
                       'target_nums': target_nums,
                       }
try:
    flag_dict = FLAGS.flag_values_dict()
except:
    print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
    flag_dict = FLAGS.__flags

# fill flag dict for plot
flag_dict['n_regular'] = n_regular
flag_dict['n_adaptive'] = n_adaptive
flag_dict['recall_cue'] = True

full_path = os.path.join(result_folder, file_reference)
if not os.path.exists(full_path):
    os.makedirs(full_path)

save_file(flag_dict, full_path, 'flag', file_type='json')


# training loop
t_train = 0
t_ref = time()
for k_iter in range(FLAGS.n_iter):
    # Monitor the training with a validation set
    if np.mod(k_iter, FLAGS.validate_every) == 0:
        t0 = time()
        val_dict = get_data_dict(FLAGS.n_batch)
        results_values = sess.run(results_tensors, feed_dict=val_dict)
        validation_loss_list.append(results_values['loss_recall'])
        validation_error_list.append(results_values['recall_errors'])
        t_run = time() - t0

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

        if FLAGS.do_plot:
            plot_result_tensors['out_plot'] = out_plot
            plot_result_tensors['y_predict'] = y_predict
            plot_result_tensors['thr'] = FLAGS.thr + b * beta
            if FLAGS.eprop_impl == 'hardcoded':
                plot_result_tensors['e_trace'] = e_trace
                plot_result_tensors['epsilon_a'] = epsilon_a
            plot_results_values = sess.run(plot_result_tensors, feed_dict=val_dict)
            plot_results_values['flags'] = flag_dict
            plot_trace = True if FLAGS.eprop_impl == 'hardcoded' else False
            update_plot(plot_results_values, ax_list, plot_traces=plot_trace, n_max_neuron_per_raster=20,
                        title='Training at iteration ' + str(k_iter))

            tmp_path = os.path.join(full_path,
                                    'figure' + start_time.strftime("%H%M") + '_' + '_' + 'iter_' + str(k_iter) + '.pdf')
            fig.savefig(tmp_path, format='pdf')
            plt.draw()
            plt.pause(1)

    # do early stopping check if single batch validation error under stop_crit
    if (k_iter > 0 and validation_error_list[-1] < FLAGS.stop_crit):
        early_stopping_list = []
        t_es_0 = time()
        for i in range(8):
            val_dict = get_data_dict(FLAGS.n_batch)
            early_stopping_list.append(sess.run(results_tensors['recall_errors'], feed_dict=val_dict))
        t_es = time() - t_es_0
        print("comput. time (s): early stopping: " + str(t_es))
        if np.mean(early_stopping_list) < FLAGS.stop_crit:
            n_iter_list.append(k_iter)
            print('less than ' + str(FLAGS.stop_crit) + ' - stopping training at iteration ' + str(k_iter))
            break
        else:
            print('early stopping error: ' + str(np.mean(early_stopping_list)) + ' higher than stop crit of: '
                  + str(FLAGS.stop_crit) + ' CONTINUE TRAINING')

    if k_iter == FLAGS.n_iter - 1:
        n_iter_list.append(k_iter)
        break

    # do train step
    train_dict = get_data_dict(FLAGS.n_batch)
    t0 = time()
    sess.run(train_step, feed_dict=train_dict)
    t_train = time() - t0
    training_time_list.append(t_train)

print('FINISHED IN {:.2g} s'.format(time() - t_ref))

# Save training progress
results = {
    'iterations': n_iter_list,
    'final_loss': validation_loss_list[-1],
    'val_errors': validation_error_list,
    'val_losses': validation_loss_list,
    'training_time': training_time_list,
    'flags': flag_dict,
}

save_file(results, full_path, 'training_results', file_type='json')
# Save sample trajectory (input, output, etc. for plotting) and test final performance
test_errors = []
for i in range(4):
    test_dict = get_data_dict(FLAGS.n_batch)
    results_values, plot_results_values, in_spk, spk, target_nums_np = sess.run(
        [results_tensors, plot_result_tensors, input_spikes, z, target_nums],
        feed_dict=test_dict)
    test_errors.append(results_values['recall_errors'])
    flag_dict['n_regular'] = n_regular
    plot_results_values['flags'] = flag_dict

    if FLAGS.do_plot:
        update_plot(plot_results_values, ax_list, n_max_neuron_per_raster=20,
                    title='Training at iteration ' + str(k_iter))
        tmp_path = os.path.join(full_path,
                                'figure_test' + start_time.strftime("%H%M") + '_' +
                                str(i) + '.pdf')
        fig.savefig(tmp_path, format='pdf')
        plt.close(fig)

print('''Statistics on the test set average error {:.2g} +- {:.2g} (averaged over 16 test batches of size {})'''
      .format(np.mean(test_errors), np.std(test_errors), FLAGS.n_batch))

del sess
