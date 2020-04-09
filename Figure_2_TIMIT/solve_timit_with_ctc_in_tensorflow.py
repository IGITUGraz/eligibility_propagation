# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass

import tensorflow as tf
import numpy as np
import numpy.random as rd

from toolbox.file_saver_dumper_no_h5py import NumpyAwareEncoder
from toolbox.matplotlib_extension import strip_right_top_axis
from lstm_eprop_model import CustomLSTM
from alif_eligibility_propagation import CustomALIF, exp_convolve
from tools import TimitDataset, einsum_bij_jk_to_bik, pad_vector
import time
import matplotlib.pyplot as plt
import os
import json
from tools import folder_reference

FLAGS = tf.app.flags.FLAGS

# Accessible parameter from the shell
tf.app.flags.DEFINE_string('model', 'lstm', '[lstm, lsnn]')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer type [sgd, adam, rmsprop]')
tf.app.flags.DEFINE_string('load', '', 'Path to saved model to restore')
tf.app.flags.DEFINE_string('comment', '560_DEBUG', 'comment string added to output directory')
tf.app.flags.DEFINE_string('dataset', '../datasets/timit_processed', 'Path to dataset to use')
tf.app.flags.DEFINE_string('preproc', 'htk', 'Input preprocessing: fbank, mfccs, cochspec, cochspike, htk')
tf.app.flags.DEFINE_string('eprop', None, 'options: [None, symmetric, adaptive, random], None means use BPTT')

tf.app.flags.DEFINE_bool('plot', False, 'seed number')
tf.app.flags.DEFINE_bool('reduced_phns', False, 'Train on reduced phone set?'
                                                'Otherwise train on full set and test on reduced set.')
tf.app.flags.DEFINE_bool('verbose', True, 'Print firing rate statistics')
tf.app.flags.DEFINE_bool('peephole', True, 'Use peephole in LSTM cell when training with BPTT')
tf.app.flags.DEFINE_bool('loss_from_all_layers', False, '')
tf.app.flags.DEFINE_float('readout_decay', 1e-4, 'Decay readout [and broadcast] weights')
#
tf.app.flags.DEFINE_integer('seed', -1, 'seed number')
tf.app.flags.DEFINE_integer('n_epochs', 60, 'number of iteration ')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of layers')
tf.app.flags.DEFINE_integer('n_neuron', 250, 'number of lstm cells')
tf.app.flags.DEFINE_integer('print_every', -1, 'print every and store accuracy')
tf.app.flags.DEFINE_integer('lr_decay_every', 15, 'Decay every epochs')
tf.app.flags.DEFINE_integer('batch', 8, 'mini_batch size')
tf.app.flags.DEFINE_integer('test_batch', 64, 'mini_batch size for validation and testing')
tf.app.flags.DEFINE_integer('n_repeat', 1, 'repeat input (for lsnn model)')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_float('reg', 0, 'regularization coefficient [1e-3]')
#
tf.app.flags.DEFINE_float('n_split', 0.5, 'part of LIF neurons')
tf.app.flags.DEFINE_float('init_scale', 0.1, '')
tf.app.flags.DEFINE_float('l2', 0., 'L2 regularization')
tf.app.flags.DEFINE_float('lr_decay', .3, 'learning rate decay factor')
tf.app.flags.DEFINE_float('lr_init', 0.001, '')
tf.app.flags.DEFINE_float('adam_epsilon', 1e-5, '')
tf.app.flags.DEFINE_float('momentum', 0.9, '')
tf.app.flags.DEFINE_float('input_noise', 0.0, 'std of input noise')
tf.app.flags.DEFINE_float('gd_noise', 0.0, 'std of gradient noise')
tf.app.flags.DEFINE_float('w_noise', 0.0, 'std of weight noise')
tf.app.flags.DEFINE_float('noise_epoch_start', 0, 'after how many epochs to start adding noise')
tf.app.flags.DEFINE_float('drop_out_probability', 0.7, 'keep probability of dropout regularization')
tf.app.flags.DEFINE_float('grad_clip', 1, 'gradient clipping global norm')
# LSNN parameters
tf.app.flags.DEFINE_float('dt', 1., 'Simulation discrete time step in ms')
tf.app.flags.DEFINE_float('thr', 0.01, 'Baseline threshold voltage')
tf.app.flags.DEFINE_float('tau_a', 500, 'Adaptation time constant')
tf.app.flags.DEFINE_float('tau_v', 20, 'Membrane time constant of recurrent neurons')
tf.app.flags.DEFINE_float('beta', 1.8, 'Scaling constant of the adaptive threshold')
tf.app.flags.DEFINE_float('tau_out', 3, 'time constant for PSP decay at the network output')
tf.app.flags.DEFINE_integer('n_delay', 2, 'Maximum synaptic delay in steps')
tf.app.flags.DEFINE_integer('n_ref', 2, 'Number of refractory steps')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Parameter necessary to approximate the spike derivative')

key0 = list(dir(FLAGS))[0]
getattr(FLAGS, key0)

assert FLAGS.eprop in [None, 'symmetric', 'adaptive', 'random']

if FLAGS.model == 'lsnn':
    FLAGS.n_repeat = 5
    FLAGS.n_neuron = 800
    FLAGS.n_split = 0.1

if FLAGS.eprop is not None:
    assert FLAGS.readout_decay is not None
    FLAGS.loss_from_all_layers = True
    FLAGS.peephole = False

n_lif = int(FLAGS.n_neuron * FLAGS.n_split)
n_alif = FLAGS.n_neuron - n_lif

script_name = os.path.basename(__file__)[:-3]
folder_reference = folder_reference(script_name, FLAGS)
#
LstmCell = tf.nn.rnn_cell.LSTMCell

# After processing the data, this object loads it and prepare it.
dataset = TimitDataset(FLAGS.batch, data_path=FLAGS.dataset, preproc=FLAGS.preproc,
                       return_sparse_phonem_tensor=True, use_reduced_phonem_set=FLAGS.reduced_phns)

# Placeholders loaded from data
batch_size = tf.Variable(0, dtype=tf.int32, trainable=False, name="BatchSize")
audio = tf.placeholder(shape=(None,None),dtype=tf.float32,name='Audio')
features = tf.placeholder(shape=(None, None, dataset.n_features), dtype=tf.float32, name='Features')
phns = tf.sparse_placeholder(dtype=tf.int32)
seq_len = tf.placeholder(dtype=tf.int32, shape=[None])
weighted_relevant_mask = tf.placeholder(shape=(None, None), dtype=tf.float32, name="RelevanceMask")
keep_prob = tf.placeholder(dtype=tf.float32, shape=())
noise_gate = tf.placeholder(dtype=tf.float32, shape=())

features = features + noise_gate * tf.random_normal(stddev=FLAGS.input_noise, shape=tf.shape(features))

# Non-trainable variables that are used to implement a decaying learning rate and count the iterations
lr = tf.Variable(FLAGS.lr_init, dtype=tf.float32, trainable=False)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
gd_noise = tf.Variable(0, dtype=tf.float32, trainable=False)
lr_update = tf.assign(lr, lr * FLAGS.lr_decay)


def batch_to_feed_dict(batch, is_train, add_noise=False):
    '''
    Create the dictionnary that is fed into the Session.run(..) calls.
    :param batch:
    :return:
    '''
    features_np, phns_np, seq_len_np, wav_np = batch
    n_time = max([len(i) for i in wav_np])
    wav_np = np.stack([pad_vector(w, n_time) for w in wav_np], axis=0)

    n_batch, n_time, n_features = features_np.shape
    relevance_mask_np = [(np.arange(n_time) < seq_len_np[i]) / seq_len_np[i] for i in range(n_batch)]
    relevance_mask_np = np.array(relevance_mask_np)

    if FLAGS.n_repeat > 1:
        # Extend sequences with the repeat in time
        features_np = np.repeat(features_np, FLAGS.n_repeat, axis=1)

    n_batch, n_time, n_features = features_np.shape
    phns_labels = tf.SparseTensorValue(phns_np['indices'], phns_np['values'], [n_batch, n_time])

    return {features: features_np, phns: phns_labels, seq_len: seq_len_np, weighted_relevant_mask: relevance_mask_np,
            keep_prob: FLAGS.drop_out_probability if is_train else 1., batch_size: n_batch,
            noise_gate: 1. if is_train else 0., audio: wav_np, gd_noise: FLAGS.gd_noise if add_noise else 0.}


initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale) if FLAGS.init_scale > 0 else None


def get_cell(n_neuron, tag, n_input):
    if FLAGS.model == 'lstm':
        return CustomLSTM(n_neuron, use_peepholes=FLAGS.peephole, stop_gradients=FLAGS.eprop is not None)
    if FLAGS.model == 'lsnn':
        thr_new = FLAGS.thr / (1 - np.exp(-FLAGS.dt / FLAGS.tau_v))
        beta_new = FLAGS.beta * (1 - np.exp(-FLAGS.dt / FLAGS.tau_a)) / (1 - np.exp(-FLAGS.dt / FLAGS.tau_v))
        beta = np.concatenate([np.zeros(n_lif), np.ones(n_alif) * beta_new])
        print("for CustomALIF new threshold = {:.4g}\nfor CustomALIF new beta      = {:.4g}".format(thr_new, beta_new))
        return CustomALIF(n_in=n_input, n_rec=n_lif + n_alif, tau=FLAGS.tau_v,
                          dt=FLAGS.dt, tau_adaptation=FLAGS.tau_a, beta=beta, thr=thr_new,
                          dampening_factor=FLAGS.dampening_factor,
                          tag=tag, n_refractory=FLAGS.n_ref,
                          stop_gradients=FLAGS.eprop is not None,
                          )
    else:
        raise NotImplementedError("Unknown model: " + FLAGS.model)


# Define the graph for the RNNs processing
with tf.variable_scope('RNNs', initializer=initializer):
    inputs = features


    def bi_directional_lstm(inputs, n_neuron, layer_number):

        if FLAGS.drop_out_probability > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

        with tf.variable_scope('BiDirectionalLayer' + str(layer_number), initializer=initializer):
            cell_forward = get_cell(n_neuron, "fw", dataset.n_features if layer_number == 0 else n_neuron*2)
            cell_backward = get_cell(n_neuron, "bw", dataset.n_features if layer_number == 0 else n_neuron*2)
            print(type(cell_forward).__name__)

            if FLAGS.model == 'lstm':
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_forward, cell_backward, inputs=inputs, dtype=tf.float32, sequence_length=seq_len)
                outputs_forward, outputs_backward = outputs
            elif FLAGS.model == 'lsnn':
                outputs_forward, _ = tf.nn.dynamic_rnn(cell_forward, inputs,
                                                       dtype=tf.float32, scope='fwRNN', swap_memory=True)
                outputs_backward, _ = tf.nn.dynamic_rnn(cell_backward, tf.reverse(inputs, axis=[1]),
                                                        dtype=tf.float32, scope='bwRNN', swap_memory=True)

                outputs_forward, _, _, _ = outputs_forward
                outputs_backward, _, _, _ = outputs_backward
                outputs_backward = tf.reverse(outputs_backward, axis=[1])

            outputs = tf.concat([outputs_forward, outputs_backward], axis=2)
            return outputs


    def multi_rnn_cells(inputs, n_neuron, n_layer):
        output_list = []
        for k_layer in range(n_layer):
            outputs = bi_directional_lstm(inputs, n_neuron, k_layer)
            output_list.append(outputs)
            inputs = outputs
        return output_list


    output_list = multi_rnn_cells(inputs, FLAGS.n_neuron, FLAGS.n_layer)

    if FLAGS.loss_from_all_layers:
        outputs = tf.concat(output_list, axis=2)
        n_outputs = FLAGS.n_neuron * 2 * FLAGS.n_layer
    else:
        outputs = output_list[-1]
        n_outputs = FLAGS.n_neuron * 2

if FLAGS.drop_out_probability > 0:
    outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)

if FLAGS.model == 'lsnn':
    outputs = exp_convolve(outputs, decay=np.exp(-FLAGS.dt / FLAGS.tau_out))

if FLAGS.n_repeat > 1:
    n_neurons = (n_lif + n_alif) * 2
    new_shape = tf.convert_to_tensor((batch_size, -1, FLAGS.n_repeat, n_outputs), dtype=tf.int32)
    outputs = tf.reshape(outputs, shape=new_shape)
    outputs = tf.reduce_mean(outputs, axis=2)


@tf.custom_gradient
def BA_logits(psp, W_out, BA_out):
    logits = einsum_bij_jk_to_bik(psp, W_out)
    def grad(dy):
        dloss_dw_out = tf.einsum('btj,btk->jk', psp, dy)
        dloss_dba_out = tf.einsum('btj,btk->jk', psp, dy) if FLAGS.eprop == 'adaptive' else tf.zeros_like(BA_out)
        dloss_dpsp = tf.einsum('bik,jk->bij', dy, BA_out)
        return [dloss_dpsp, dloss_dw_out, dloss_dba_out]
    return logits, grad


# Define the graph for the output processing
with tf.name_scope('Output'):
    N_output_classes_with_blank = dataset.n_phns + 1
    print('Output classes {} + 1 for ctc blank'.format(N_output_classes_with_blank - 1))
    w_out = tf.Variable(rd.randn(n_outputs, N_output_classes_with_blank) / np.sqrt(n_outputs), dtype=tf.float32)
    b_out = tf.Variable(np.zeros(N_output_classes_with_blank), dtype=tf.float32)

    if FLAGS.eprop in ['adaptive', 'random']:
        if FLAGS.eprop == 'adaptive':
            init_w_out = tf.constant(rd.randn(n_outputs, N_output_classes_with_blank) / np.sqrt(n_outputs), dtype=tf.float32)
            BA_out = tf.Variable(init_w_out, dtype=tf.float32, name='BroadcastWeights')
        else:
            init_w_out = tf.constant(rd.randn(n_outputs, N_output_classes_with_blank), dtype=tf.float32)
            BA_out = tf.constant(init_w_out, dtype=tf.float32, name='BroadcastWeights')
        phn_logits = BA_logits(outputs, w_out, BA_out) + b_out
    else:
        phn_logits = einsum_bij_jk_to_bik(outputs, w_out) + b_out

if FLAGS.eprop == 'adaptive':
    weight_decay = tf.constant(FLAGS.readout_decay, dtype=tf.float32)
    w_out_decay = tf.assign(w_out, w_out - weight_decay * w_out)
    BA_decay = tf.assign(BA_out, BA_out - weight_decay * BA_out)
    KolenPollackDecay = [BA_decay, w_out_decay]

# Firing rate regularization
with tf.name_scope('RegularizationLoss'):
    av = tf.reduce_mean(tf.concat(output_list, axis=2), axis=(0, 1)) / FLAGS.dt
    loss_reg = tf.reduce_sum(tf.square(av - FLAGS.reg_rate / 1000) * FLAGS.reg)

# Define the graph for the loss function and the definition of the error
with tf.name_scope('Loss'):
    phn_logits_time_major = tf.transpose(phn_logits, [1, 0, 2])

    negative_loss_probs = tf.nn.ctc_loss(phns, phn_logits_time_major, seq_len, )
    loss = tf.reduce_mean(negative_loss_probs)
    loss = tf.reduce_mean(loss)

    loss = loss + loss_reg

    if FLAGS.l2 > 0:
        losses_l2 = [tf.reduce_sum(tf.square(w)) for w in tf.trainable_variables()]
        loss += FLAGS.l2 * tf.reduce_sum(losses_l2)

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(phn_logits_time_major, seq_len, beam_width=100, top_paths=1)

    phn_map = tf.constant(dataset.phonem_reduction_map, dtype=tf.int32)
    get_reduced_phn_id = lambda k: phn_map[k]


    def get_reduced_phn_sparse_tensor_path(path):
        path = tf.cast(path, tf.int32)
        if FLAGS.reduced_phns:
            return path
        reduce_phn_path_value = tf.map_fn(get_reduced_phn_id, path.values)

        reduced_decoded_path = tf.SparseTensorValue(
            path.indices,
            reduce_phn_path_value,
            path.dense_shape)
        return reduced_decoded_path


    ler = tf.reduce_mean(
        tf.edit_distance(get_reduced_phn_sparse_tensor_path(decoded[0]), get_reduced_phn_sparse_tensor_path(phns)))


# Define the training step operation
with tf.name_scope('Train'):
    var_list = tf.trainable_variables()
    [print(v) for v in var_list]

    if FLAGS.optimizer == 'sgd':
        opt = tf.train.MomentumOptimizer(lr, momentum=FLAGS.momentum)
    elif FLAGS.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=FLAGS.adam_epsilon, beta1=FLAGS.momentum)
    elif FLAGS.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(lr, epsilon=FLAGS.adam_epsilon, momentum=FLAGS.momentum)

    gradients, variables = zip(*opt.compute_gradients(loss))
    if FLAGS.grad_clip > 0: gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.grad_clip)

    get_noise = lambda var: tf.random_normal(shape=tf.shape(var), stddev=gd_noise)
    grads_vars = [(g + get_noise(v),v) for g, v in zip(gradients,variables)]

    train_step = opt.apply_gradients(grads_vars, global_step=global_step)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if FLAGS.load:
    saver.restore(sess, FLAGS.load)

if FLAGS.plot:
    plt.ion()
    fig, ax_list = plt.subplots(2, figsize=(8, 4))


def update_plot(i=0):
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)
    result_plot_tensors = {'features': features, 'phns': phns, 'phn_predictions': decoded, 'phn_logits': phn_logits}
    plot_feed_dict = batch_to_feed_dict(dataset.get_validation_batch(), is_train=True)
    result_plot_values = sess.run(result_plot_tensors, feed_dict=plot_feed_dict)

    txt = dataset.meta_data_train[i]['text']

    def sparse_tensor_to_string(sp_phn_tensor, i_batch):
        selection = sp_phn_tensor.indices[:, 0] == i_batch
        phn_list = sp_phn_tensor.values[selection]
        str_phn_list = [dataset.vocabulary[k] for k in phn_list]
        str_phn_list = ['_' if phn == 'sil' else phn for phn in str_phn_list]
        return ' '.join(str_phn_list)

    seq_len = np.argmax((result_plot_values['features'][0] == 0.).all(axis=1))

    ax_list[0].set_title(txt)
    ax_list[0].imshow(result_plot_values['features'][0].T)
    ax_list[0].set_ylabel('Audio features')
    ax_list[0].set_xticklabels([])

    str_1 = sparse_tensor_to_string(sp_phn_tensor=result_plot_values['phns'], i_batch=0)
    str_2 = sparse_tensor_to_string(sp_phn_tensor=result_plot_values['phn_predictions'][0], i_batch=0)
    ax_list[0].set_xlabel('frames' '\n' + 'target:' + str_1 + '\n' + 'prediciton:' + str_2)

    ax_list[1].imshow(result_plot_values['phn_logits'][0].T, aspect="auto")
    ax_list[1].set_ylabel('logits')
    ax_list[1].set_xlabel('steps')

    fig.subplots_adjust(hspace=0.2)
    for ax in ax_list:
        ax.set_xlim([0, seq_len])
        ax.grid(color='black', alpha=0.4, linewidth=0.4)

    plt.draw()
    plt.pause(1)


results = {
    'loss_list': [],
    'ler_list': [],
    'ler_test_list': [],
    'n_synapse': [],
    'iteration_list': [],
    'epoch_list': [],
    'training_time_list': [],
    'av_list': [],
    'fr_max_list': [],
    'fr_avg_list': [],
}

training_time = 0
testing_time = 0
test_result_tensors = {
    'ler': ler,
    'loss': loss,
    'av': av,
}
train_result_tensors = {'train_step': train_step}

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print("NUM OF PARAMETERS = ", total_parameters)
results['total_parameters'] = total_parameters

epoch_last_iteration = -1
best_loss = np.inf


def add_random_noise(w, mean=0.0, stddev=FLAGS.w_noise):
    variables_shape = tf.shape(w)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    return tf.assign_add(w, noise)


def compute_result(type="validation"):
    assert type in ["validation", "test"]
    total_batch_size = dataset.n_develop if type == "validation" else dataset.n_test
    n_minibatch = total_batch_size // FLAGS.test_batch
    mini_batch_sizes = [FLAGS.test_batch for _ in range(n_minibatch)]
    if total_batch_size - (n_minibatch * FLAGS.test_batch) != 0:
        mini_batch_sizes = mini_batch_sizes + [total_batch_size - (n_minibatch * FLAGS.test_batch)]

    # collect_results = dict.fromkeys(test_result_tensors.keys(), list())
    collect_results = {k: [] for k in test_result_tensors.keys()}
    for idx, mb_size in enumerate(mini_batch_sizes):
        selection = np.arange(mb_size)
        selection = selection + np.ones_like(selection) * idx * FLAGS.test_batch

        if type == "validation":
            data = dataset.get_next_validation_batch(selection)
        elif type == "test":
            data = dataset.get_next_test_batch(selection)

        feed_dict = batch_to_feed_dict(data, is_train=False)
        run_output = sess.run(test_result_tensors, feed_dict=feed_dict)
        for k, value in run_output.items():
            collect_results[k].append(value)

    mean_result = {key: np.mean(collect_results[key], axis=0) for key in collect_results.keys()}
    return mean_result


while dataset.current_epoch <= FLAGS.n_epochs:
    k_iteration = sess.run(global_step)
    is_new_epoch = epoch_last_iteration == dataset.current_epoch - 1

    if FLAGS.lr_decay_every > 0 and np.mod(dataset.current_epoch, FLAGS.lr_decay_every) == 0 and is_new_epoch and k_iteration > 0:
        sess.run(lr_update)
        print('Decay learning rate: {:.2g}'.format(sess.run(lr)))

    if is_new_epoch or (FLAGS.print_every > 0 and np.mod(k_iteration,FLAGS.print_every) == 0 and k_iteration > 0):
        t0 = time.time()
        test_result = compute_result("test")
        valid_result = compute_result("validation")
        testing_time = time.time() - t0

        print('Epoch: {} time/it: {:.2f} s it: {}  \t PER: {:.3g} (valid) \t '
              'loss {:.3g} (valid) \t PER: {:.3g} (test)'.format(
                dataset.current_epoch, training_time, k_iteration, valid_result['ler'],
                valid_result['loss'], test_result['ler']))


        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return min_val, max_val, np.mean(v), np.std(v), k_min, k_max


        firing_rate_stats = get_stats(valid_result['av'] * 1000)

        if FLAGS.verbose and FLAGS.model == 'lsnn':
            print('firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f}'.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3]
            ))

        if FLAGS.plot:
            update_plot()

        if valid_result['loss'] < best_loss:
            saver.save(sess, os.path.join(folder_reference, 'best_model'))

        for key, value in valid_result.items():
            results[key + '_list'].append(value)
        results['ler_test_list'].append(test_result['ler'])
        idx_min_valid = np.argmin(results['ler_list'])
        early_stopping_test_error = results['ler_test_list'][idx_min_valid]
        results['early_stopping_test'] = early_stopping_test_error

        for key, variable in zip(['iteration', 'epoch', 'training_time'],
                                 [k_iteration, dataset.current_epoch, training_time]):
            results[key + '_list'].append(variable)
        results['fr_max_list'].append(firing_rate_stats[1])
        results['fr_avg_list'].append(firing_rate_stats[2])

        with open(os.path.join(folder_reference, 'metrics.json'), 'w') as f:
            json.dump(results, f, cls=NumpyAwareEncoder, indent=2)

    #
    epoch_last_iteration = dataset.current_epoch

    t0 = time.time()
    sess.run(train_step, feed_dict=batch_to_feed_dict(dataset.get_next_training_batch(), is_train=True,
                                                      add_noise=dataset.current_epoch > FLAGS.noise_epoch_start))
    if FLAGS.eprop == 'adaptive':
        sess.run(KolenPollackDecay)
    training_time = time.time() - t0

    if FLAGS.w_noise > 0:
        for variable in tf.trainable_variables():
            sess.run(add_random_noise(variable))

del sess
