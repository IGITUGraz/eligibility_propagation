"""
    Training LSNN model to solve framewise phone classification of TIMIT dataset

    CUDA_VISIBLE_DEVICES=0 python3 -u solve_timit_with_framewise_lsnn.py
"""

import tensorflow as tf
import numpy as np
import numpy.random as rd

from alif_eligibility_propagation import CustomALIF, exp_convolve
from toolbox.matplotlib_extension import raster_plot, strip_right_top_axis
from toolbox.file_saver_dumper_no_h5py import NumpyAwareEncoder
from tools import TimitDataset, einsum_bij_jk_to_bik, pad_vector
import time
import os
import errno
import json
import datetime


def flag_to_dict(FLAG):
    if float(tf.__version__[2:]) >= 5:
        flag_dict = FLAG.flag_values_dict()
    else:
        flag_dict = FLAG.__flags
    return flag_dict


script_name = os.path.basename(__file__)[:-3]
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M__%S_%f")
try:
    os.makedirs('results')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

FLAGS = tf.app.flags.FLAGS

# Accessible parameter from the shell
tf.app.flags.DEFINE_string('comment', '', 'comment attached to output filenames')
tf.app.flags.DEFINE_string('run_id', '', 'comment attached to output filenames')
tf.app.flags.DEFINE_string('checkpoint', '', 'optionally load the pre-trained weights from checkpoint')
tf.app.flags.DEFINE_string('preproc', 'htk', 'Input preprocessing: fbank, mfccs, cochspec, cochspike, htk')
tf.app.flags.DEFINE_string('eprop', None, 'options: [None, symmetric, adaptive, random], None means use BPTT')
tf.app.flags.DEFINE_bool('adam', True, 'use Adam optimizer')
tf.app.flags.DEFINE_bool('plot', False, 'Interactive plot during training (useful for debugging)')
tf.app.flags.DEFINE_bool('reduced_phns', False, 'Use reduced phone set')
tf.app.flags.DEFINE_bool('psp_out', True, 'Use accumulated PSP instead of raw spikes of model as output')
tf.app.flags.DEFINE_bool('verbose', True, '')
tf.app.flags.DEFINE_bool('ramping_learning_rate', True, 'Ramp up the learning rate from 0 to lr_init in first epoch')
tf.app.flags.DEFINE_bool('BAglobal', False, 'Enable broadcast alignment with uniform weights to all neurons')
tf.app.flags.DEFINE_bool('cell_train', True, 'Train the RNN cell')
tf.app.flags.DEFINE_bool('readout_bias', True, 'Use bias variable in readout')
tf.app.flags.DEFINE_bool('rec', True, 'Use recurrent weights. Used to provide a baseline.')
tf.app.flags.DEFINE_string('dataset', '../datasets/timit_processed_htk_mfccs', 'Path to dataset to use')
tf.app.flags.DEFINE_float('readout_decay', 1e-2, 'Decay readout [and broadcast] weights')
tf.app.flags.DEFINE_bool('loss_from_all_layers', True, 'For multi-layer setup, make readout from all layers.')
#
tf.app.flags.DEFINE_integer('seed', -1, 'seed number')
tf.app.flags.DEFINE_integer('n_epochs', 80, 'number of iteration ')
tf.app.flags.DEFINE_integer('n_layer', 1, 'number of layers')
tf.app.flags.DEFINE_integer('n_regular', 300, 'number of regular spiking units in the recurrent layer.')
tf.app.flags.DEFINE_integer('n_adaptive', 100, 'number of adaptive spiking units in the recurrent layer')
tf.app.flags.DEFINE_integer('print_every', 100, 'print every and store accuracy')
tf.app.flags.DEFINE_integer('lr_decay_every', -1, 'Decay every')
tf.app.flags.DEFINE_integer('batch', 32, 'mini_batch size')
tf.app.flags.DEFINE_integer('test_batch', 32, 'mini_batch size')
tf.app.flags.DEFINE_integer('n_ref', 2, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('n_repeat', 5, 'repeat each input time step for this many simulation steps (ms)')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('truncT', -1, 'truncate time to this many input steps (truncT * n_repeat ms)')
#
tf.app.flags.DEFINE_float('dt', 1., 'Membrane time constant of output readouts')
tf.app.flags.DEFINE_float('tau_a', 200, 'Adaptation time constant')
tf.app.flags.DEFINE_bool('tau_a_spread', False, 'Spread time constants uniformly from 0 to tau_a')
tf.app.flags.DEFINE_float('tau_v', 20, 'Membrane time constant of recurrent neurons')
tf.app.flags.DEFINE_bool('tau_v_spread', False, 'Spread time constants uniformly from 0 to tau_v')
tf.app.flags.DEFINE_float('beta', 1.8, 'Scaling constant of the adaptive threshold')
tf.app.flags.DEFINE_float('clip', 0., 'Proportion of connected synpases at initialization')
tf.app.flags.DEFINE_float('l2', 1e-5, '')
tf.app.flags.DEFINE_float('lr_decay', .3, '')
tf.app.flags.DEFINE_float('lr_init', 0.01, '')
tf.app.flags.DEFINE_float('adam_epsilon', 1e-5, '')
tf.app.flags.DEFINE_float('momentum', 0.9, '')
tf.app.flags.DEFINE_float('gd_noise', 0.06 ** 2 * 10, 'Used only when noise_step_start > 0')
tf.app.flags.DEFINE_float('noise_step_start', -1, 'was 1000')
tf.app.flags.DEFINE_float('thr', 0.01, 'Baseline threshold voltage')
tf.app.flags.DEFINE_float('proportion_excitatory', 0.75, 'proportion of excitatory neurons')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring (irrelevant without rewiring)')
tf.app.flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Parameter necessary to approximate the spike derivative')
tf.app.flags.DEFINE_float('tau_out', 3, 'Mikolov: tau for PSP decay at output')
tf.app.flags.DEFINE_float('reg', 50, 'regularization coefficient')
tf.app.flags.DEFINE_float('drop_out_probability', -1., '')

tf.app.flags.DEFINE_integer('cuda_device', -1, '')

if FLAGS.plot:
    import matplotlib.pyplot as plt
#
key0 = list(dir(FLAGS))[0]
getattr(FLAGS, key0)

if FLAGS.cuda_device >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.cuda_device)

filename = time_stamp + '_' + FLAGS.comment + '_' + FLAGS.run_id
storage_path = os.path.join('results', script_name, filename)
print("STORING EVERYTHING TO: ", storage_path)
try:
    os.makedirs(storage_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

if FLAGS.n_repeat < 1:
    FLAGS.n_repeat = 1
flagdict = flag_to_dict(FLAGS)
assert isinstance(flagdict, dict)

# After processing the data, this object loads it and prepare it.
dataset = TimitDataset(FLAGS.batch, data_path=FLAGS.dataset, preproc=FLAGS.preproc,
                       use_reduced_phonem_set=FLAGS.reduced_phns)
n_in = dataset.n_features

# Placeholders loaded from data
features = tf.placeholder(shape=(None, None, dataset.n_features), dtype=tf.float32, name='Features')
audio = tf.placeholder(shape=(None, None), dtype=tf.float32, name='Audio')
phns = tf.placeholder(shape=(None, None), dtype=tf.int64, name='Labels')
seq_len = tf.placeholder(dtype=tf.int32, shape=[None], name="SeqLen")
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="KeepProb")
weighted_relevant_mask = tf.placeholder(shape=(None, None), dtype=tf.float32, name="RelevanceMask")

batch_size = tf.Variable(0, dtype=tf.int32, trainable=False, name="BatchSize")

# Non-trainable variables that are used to implement a decaying learning rate and count the iterations
lr = tf.Variable(FLAGS.lr_init, dtype=tf.float32, trainable=False, name="LearningRate")
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="GlobalStep")
lr_update = tf.assign(lr, lr * FLAGS.lr_decay)
gd_noise = tf.Variable(0, dtype=tf.float32, trainable=False, name="GDNoise")

# Op to ramping learning rate
n_iteration_per_epoch = 100
ramping_learning_rate_values = tf.linspace(0., 1., num=n_iteration_per_epoch)
clipped_global_step = tf.minimum(global_step, n_iteration_per_epoch - 1)
ramping_learning_rate_op = tf.assign(lr, FLAGS.lr_init * ramping_learning_rate_values[clipped_global_step])

# Frequencies
regularization_f0 = FLAGS.reg_rate / 1000


def batch_to_feed_dict(batch, is_train):
    '''
    Create the dictionnary that is fed into the Session.run(..) calls.
    :param batch:
    :return:
    '''
    features_np, phns_np, seq_len_np, wav_np = batch

    n_time = max([len(i) for i in wav_np])
    wav_np = np.stack([pad_vector(w, n_time) for w in wav_np], axis=0)

    # print("input max ", np.max(features_np))
    n_batch, n_time, n_features = features_np.shape
    relevance_mask_np = [(np.arange(n_time) < seq_len_np[i]) / seq_len_np[i] for i in range(n_batch)]
    relevance_mask_np = np.array(relevance_mask_np)
    if FLAGS.n_repeat > 1:
        # Extend sequences with the repeat in time
        features_np = np.repeat(features_np, FLAGS.n_repeat, axis=1)
        seq_len_np *= FLAGS.n_repeat

    if FLAGS.truncT > 0 and is_train:
        in_steps_len = phns_np.shape[1]
        if in_steps_len <= FLAGS.truncT:
            print("truncT (", FLAGS.truncT, ") too long! setting to smaller size found = ", in_steps_len - 1)
            FLAGS.truncT = in_steps_len - 1
        max_step_offset = in_steps_len - FLAGS.truncT
        rnd_step_offset = rd.randint(low=0, high=max_step_offset)
        features_np = features_np[:, rnd_step_offset * FLAGS.n_repeat:(rnd_step_offset + FLAGS.truncT) * FLAGS.n_repeat,
                      :]
        phns_np = phns_np[:, rnd_step_offset:rnd_step_offset + FLAGS.truncT]
        seq_len_np = np.array(seq_len_np)
        seq_len_np[seq_len_np > FLAGS.truncT] = FLAGS.truncT

        relevance_mask_np = relevance_mask_np[:, rnd_step_offset:rnd_step_offset + FLAGS.truncT]

    n_batch, n_time, n_features = features_np.shape
    phns_labels = phns_np
    return {features: features_np, phns: phns_labels, seq_len: seq_len_np, weighted_relevant_mask: relevance_mask_np,
            batch_size: n_batch, keep_prob: FLAGS.drop_out_probability if is_train else 1., audio: wav_np}


if FLAGS.tau_a_spread:
    taua = rd.choice([1, 0.5], size=FLAGS.n_regular + FLAGS.n_adaptive) * FLAGS.tau_a
else:
    taua = FLAGS.tau_a
if FLAGS.tau_v_spread:
    tauv = rd.choice([1, 0.5], size=FLAGS.n_regular + FLAGS.n_adaptive) * FLAGS.tau_v
else:
    tauv = FLAGS.tau_v

flagdict['tauas'] = taua.tolist() if type(taua) is not float else taua
flagdict['tauvs'] = tauv.tolist() if type(tauv) is not float else tauv
with open(os.path.join(storage_path, 'flags.json'), 'w') as f:
    json.dump(flagdict, f, indent=2)


def get_cell(tag, n_input=n_in):
    # converting thr and beta parameters
    thr_new = FLAGS.thr / (1 - np.exp(-FLAGS.dt / tauv)) if np.isscalar(tauv) else \
        [FLAGS.thr / (1 - np.exp(-FLAGS.dt / tv)) for tv in tauv]
    if np.isscalar(tauv) and np.isscalar(taua):
        beta_new = FLAGS.beta * (1 - np.exp(-FLAGS.dt / taua)) / (1 - np.exp(-FLAGS.dt / tauv))
        print("565 new threshold = {:.4g}\n565 new beta      = {:.4g}".format(thr_new, beta_new))
        beta_new = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * beta_new])
    elif np.isscalar(tauv) and not np.isscalar(taua):
        beta_new = np.array([FLAGS.beta * (1 - np.exp(-FLAGS.dt / ta)) / (1 - np.exp(-FLAGS.dt / tauv)) for ta in taua])
        beta_new[:FLAGS.n_regular] = 0
    elif not np.isscalar(tauv) and np.isscalar(taua):
        beta_new = np.array([FLAGS.beta * (1 - np.exp(-FLAGS.dt / taua)) / (1 - np.exp(-FLAGS.dt / tv)) for tv in tauv])
        beta_new[:FLAGS.n_regular] = 0
    elif not np.isscalar(tauv) and not np.isscalar(taua):
        beta_new = np.array(
            [FLAGS.beta * (1 - np.exp(-FLAGS.dt / ta)) / (1 - np.exp(-FLAGS.dt / tv)) for ta, tv in zip(taua, tauv)])
        beta_new[:FLAGS.n_regular] = 0
    else:
        raise NotImplementedError("Nonexistant combination of taua tauv")

    return CustomALIF(n_in=n_input, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tauv,
                      dt=FLAGS.dt, tau_adaptation=taua, beta=beta_new, thr=thr_new,
                      dampening_factor=FLAGS.dampening_factor,
                      tag=tag, n_refractory=FLAGS.n_ref,
                      stop_gradients=FLAGS.eprop is not None, rec=FLAGS.rec
                      )


# Cell model used to solve the task, we have two because we used a bi-directional network
cell_forward = get_cell("FW")
cell_backward = get_cell("BW")

# Define the graph for the RNNs processing
with tf.variable_scope('RNNs'):
    def bi_directional_lstm(inputs, layer_number):

        if FLAGS.drop_out_probability > 0:
            inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

        with tf.variable_scope('BiDirectionalLayer' + str(layer_number)):
            if layer_number == 0:
                cell_f = cell_forward
                cell_b = cell_backward
            else:
                cell_f = get_cell("FW" + str(layer_number), n_input=2 * (FLAGS.n_regular + FLAGS.n_adaptive))
                cell_b = get_cell("BW" + str(layer_number), n_input=2 * (FLAGS.n_regular + FLAGS.n_adaptive))

            outputs_forward, _ = tf.nn.dynamic_rnn(cell_f, inputs, dtype=tf.float32, scope='ForwardRNN')
            outputs_backward, _ = tf.nn.dynamic_rnn(cell_b, tf.reverse(inputs, axis=[1]), dtype=tf.float32,
                                                    scope='BackwardRNN')

            outputs_forward, _, _, _ = outputs_forward
            outputs_backward, _, _, _ = outputs_backward

            outputs_backward = tf.reverse(outputs_backward, axis=[1])
            outputs = tf.concat([outputs_forward, outputs_backward], axis=2)

            return outputs


    inputs = features
    output_list = []
    for k_layer in range(FLAGS.n_layer):
        outputs = bi_directional_lstm(inputs, k_layer)
        output_list.append(outputs)
        inputs = outputs

    if FLAGS.loss_from_all_layers:
        outputs = tf.concat(output_list, axis=2)
        n_outputs = (FLAGS.n_regular + FLAGS.n_adaptive) * 2 * FLAGS.n_layer
    else:
        outputs = output_list[-1]
        n_outputs = (FLAGS.n_regular + FLAGS.n_adaptive) * 2

if FLAGS.drop_out_probability > 0:
    outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)


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
    n_neurons = (FLAGS.n_regular + FLAGS.n_adaptive) * 2
    if FLAGS.n_repeat > 1:
        new_shape = tf.convert_to_tensor((batch_size, -1, FLAGS.n_repeat, n_outputs), dtype=tf.int32)
        outputs_ds = tf.reshape(outputs, shape=new_shape)
        outputs_ds = tf.reduce_mean(outputs_ds, axis=2)
    else:
        outputs_ds = outputs
    psp_decay_new = FLAGS.psp_out * (1 - np.exp(-FLAGS.dt / taua)) / (1 - np.exp(-FLAGS.dt / tauv))
    print("566 psp readout decay = {:.4g}".format(psp_decay_new))
    lsnn_out = exp_convolve(outputs_ds, decay=np.exp(-FLAGS.dt / FLAGS.tau_out)) if FLAGS.psp_out else outputs_ds
    N_output_classes_with_blank = dataset.n_phns + 1
    w_out = tf.Variable(rd.randn(n_outputs, N_output_classes_with_blank) / np.sqrt(n_outputs),
                        dtype=tf.float32, name="OutWeights")

    if FLAGS.eprop in ['adaptive', 'random']:
        if FLAGS.BAglobal:
            BA_out = tf.constant(np.ones((n_outputs, N_output_classes_with_blank)) / np.sqrt(n_outputs),
                                 dtype=tf.float32, name='BroadcastWeights')
        else:
            if FLAGS.eprop == 'adaptive':
                init_w_out = rd.randn(n_outputs, N_output_classes_with_blank) / np.sqrt(n_outputs)
                BA_out = tf.Variable(init_w_out, dtype=tf.float32, name='BroadcastWeights')
            else:
                init_w_out = rd.randn(n_outputs, N_output_classes_with_blank)
                BA_out = tf.constant(init_w_out, dtype=tf.float32, name='BroadcastWeights')

        phn_logits = BA_logits(lsnn_out, w_out, BA_out)
    else:
        print("Broadcast alignment disabled!")
        phn_logits = einsum_bij_jk_to_bik(lsnn_out, w_out)

    if FLAGS.readout_bias:
        b_out = tf.Variable(np.zeros(N_output_classes_with_blank), dtype=tf.float32, name="OutBias")
        phn_logits += b_out

if FLAGS.eprop == 'adaptive':
    weight_decay = tf.constant(FLAGS.readout_decay, dtype=tf.float32)
    w_out_decay = tf.assign(w_out, w_out - weight_decay * w_out)
    BA_decay = tf.assign(BA_out, BA_out - weight_decay * BA_out)
    KolenPollackDecay = [BA_decay, w_out_decay]

# Firing rate regularization
with tf.name_scope('RegularizationLoss'):
    av = tf.reduce_mean(outputs, axis=(0, 1)) / FLAGS.dt
    regularization_coeff = tf.Variable(np.ones(n_outputs) * FLAGS.reg,
                                       dtype=tf.float32, trainable=False)
    loss_reg = tf.reduce_sum(tf.square(av - regularization_f0) * regularization_coeff)

# Define the graph for the loss function and the definition of the error
with tf.name_scope('Loss'):
    loss_pred = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=phns, logits=phn_logits)
    loss_pred = tf.reduce_sum(loss_pred * weighted_relevant_mask, axis=1)
    loss_pred = tf.reduce_mean(loss_pred)

    loss = loss_pred + loss_reg

    if FLAGS.l2 > 0:
        losses_l2 = [tf.reduce_sum(tf.square(w)) for w in tf.trainable_variables()]
        loss += FLAGS.l2 * tf.reduce_sum(losses_l2)

    phn_prediction = tf.argmax(phn_logits, axis=2)
    is_correct = tf.equal(phns, phn_prediction)
    is_correct_float = tf.cast(is_correct, dtype=tf.float32)
    ler = tf.reduce_sum(is_correct_float * weighted_relevant_mask, axis=1)
    ler = 1. - tf.reduce_mean(ler)
    decoded = phn_prediction

# Define the training step operation
with tf.name_scope('Train'):
    if not FLAGS.adam:
        opt = tf.train.MomentumOptimizer(lr, momentum=FLAGS.momentum)
    else:
        opt = tf.train.AdamOptimizer(lr, epsilon=FLAGS.adam_epsilon, beta1=FLAGS.momentum)

    get_noise = lambda var: tf.random_normal(shape=tf.shape(var), stddev=gd_noise)
    grads = opt.compute_gradients(loss)

    if not FLAGS.cell_train:
        grads = [(g + get_noise(v), v) for g, v in grads if 'CustomALIF_' not in v.name]
    else:
        grads = [(g + get_noise(v), v) for g, v in grads]
    train_var_list = [var for g, var in grads]
    train_step = opt.apply_gradients(grads, global_step=global_step)
    print("NUM OF TRAINABLE", len(train_var_list))
    for v in train_var_list:
        print(v.name)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
if len(FLAGS.checkpoint) > 0:
    ckpt_vars = [v[0] for v in tf.train.list_variables(FLAGS.checkpoint[:-11])]
    var_names = [v.name for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)]
    variables_can_be_restored = [v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES) if
                                 v.name[:-2] in ckpt_vars]
    saver = tf.train.Saver(variables_can_be_restored)
    saver.restore(sess, FLAGS.checkpoint)
    print("Model restored from ", FLAGS.checkpoint)
else:
    saver = tf.train.Saver()

if FLAGS.plot:
    plt.ion()
    fig, ax_list = plt.subplots(nrows=3, figsize=(12, 6))


def sparse_tensor_to_string(sp_phn_tensor, i_batch):
    selection = sp_phn_tensor.indices[:, 0] == i_batch
    phn_list = sp_phn_tensor.values[selection]
    str_phn_list = [dataset.vocabulary[k] for k in phn_list]
    str_phn_list = ['_' if phn == 'sil' else phn for phn in str_phn_list]
    return ' '.join(str_phn_list)


def update_plot(result_plot_values):
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    txt = dataset.meta_data_develop[0]['text']

    if FLAGS.preproc == 'cochspike':
        seq_len = max([i[-1] for i in np.nonzero(result_plot_values['features'][0])])
    else:
        seq_len = np.argmax(
            (result_plot_values['features'][0] == 0.).all(axis=1)) + 50  # add 50ms to see full net activity

    ax_list[0].set_title(txt)
    if "cochspike" in FLAGS.preproc:
        raster_plot(ax_list[0], result_plot_values['features'][0])
    else:
        ax_list[0].imshow(result_plot_values['features'][0].T, aspect="auto")
    ax_list[0].set_xticklabels([])
    ax_list[0].set_ylabel('Audio features')

    i = 0  # len(dataset.meta_data_develop[0]) - 1
    ind_change = np.where(np.diff(dataset.phonem_stack_develop[i]) != 0)[0]
    phns_change = dataset.phonem_stack_develop[i][ind_change]
    if FLAGS.n_repeat > 1:
        ind_change *= FLAGS.n_repeat
    ax_list[0].set_xticks(np.concatenate([[0], ind_change]))
    tick_labels = [dataset.vocabulary[k] for k in phns_change]
    tick_labels = ['_' if lab == 'sil' else lab for lab in tick_labels]
    tick_labels.append(' ')
    ax_list[0].set_xticklabels(tick_labels)

    # raster_plot(ax_list[1], result_plot_values['outputs'][0][:, 0:(FLAGS.n_regular + FLAGS.n_adaptive):5])
    raster_plot(ax_list[1], result_plot_values['outputs'][0])
    ax_list[1].set_ylabel('LSNN\nsubsampled')
    ax_list[1].set_xticklabels([])

    logits = result_plot_values['phn_logits'][0].T
    if FLAGS.n_repeat > 1:
        logits = np.repeat(logits, repeats=FLAGS.n_repeat, axis=1)
    # print("logits shape", logits.shape)
    ax_list[2].imshow(logits, aspect="auto")
    ax_list[2].set_ylabel('logits')
    ax_list[2].set_xlabel('time in ms')

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
    'fr_max_list': [],
    'fr_avg_list': [],
}

training_time = 0
testing_time = 0
test_result_tensors = {'ler': ler,
                       'loss': loss,
                       'loss_pred': loss_pred,
                       'loss_reg': loss_reg,
                       'learning_rate': lr,
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
print("TOTAL NUM OF PARAMETERS = ", total_parameters)


def compute_result(type="validation"):
    assert type in ["validation", "test"]
    total_batch_size = dataset.n_develop if type == "validation" else dataset.n_test
    n_minibatch = total_batch_size // FLAGS.test_batch
    mini_batch_sizes = [FLAGS.test_batch for _ in range(n_minibatch)]
    if total_batch_size - (n_minibatch * FLAGS.test_batch) != 0:
        mini_batch_sizes = mini_batch_sizes + [total_batch_size - (n_minibatch * FLAGS.test_batch)]

    feed_dict = None
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

    plot_result = None
    if type == "validation":
        plot_result = sess.run(result_plot_tensors, feed_dict=feed_dict)

    mean_result = {key: np.mean(collect_results[key]) for key in collect_results.keys()}
    return mean_result, plot_result


min_valid_err = 1.
while dataset.current_epoch <= FLAGS.n_epochs:
    k_iteration = sess.run(global_step)

    if k_iteration == FLAGS.noise_step_start:
        sess.run(tf.assign(gd_noise, FLAGS.gd_noise))
        print('Setting gradient noise standard deviation to: {}'.format(sess.run(gd_noise)))

    if k_iteration < 100 and FLAGS.ramping_learning_rate:
        old_lr = sess.run(lr)
        new_lr = sess.run(ramping_learning_rate_op)
        if k_iteration == 0:
            print('Ramping learning rate during first epoch: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    if FLAGS.lr_decay_every > 0 and np.mod(k_iteration, FLAGS.lr_decay_every) == 0 and k_iteration > 0:
        sess.run(lr_update)
        print('Decay learning rate: {:.2g}'.format(sess.run(lr)))

    if np.mod(k_iteration, FLAGS.print_every) == 0:
        t0 = time.time()
        result_plot_tensors = {'features': features, 'phns': phns, 'phn_predictions': decoded,
                               'outputs': outputs, 'phn_logits': phn_logits, 'audio': audio, }

        test_result, _ = compute_result("test")
        valid_result, valid_result_plot = compute_result("validation")

        if test_result['ler'] < min_valid_err:
            min_valid_err = test_result['ler']
            saver.save(sess, os.path.join(storage_path, 'model.ckpt'))
            saver.export_meta_graph(os.path.join(storage_path, 'graph.meta'))
            # data for plotting
            for k in valid_result_plot.keys():
                valid_result_plot[k] = valid_result_plot[k][0:32]
            np.save(os.path.join(storage_path, 'valid_plot.npy'), valid_result_plot)
        testing_time = time.time() - t0

        print('Epoch: {} \t time/it: {:.2f} s \t time/test: {:.2f} \t it: {} \t '
              'PER: {:.3g} (valid) \t loss {:.3g} (valid) \t PER: {:.3g} (test)'
              .format(dataset.current_epoch, training_time, testing_time, k_iteration, valid_result['ler'],
                      valid_result['loss'], test_result['ler']))


        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max


        firing_rate_stats = get_stats(valid_result['av'] * 1000)

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f} (over neurons)
            classification loss {:.2g} \t regularization loss {:.2g} \t learning rate {:.2g}
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                valid_result['loss_pred'], valid_result['loss_reg'],
                valid_result['learning_rate']
            ))

        if FLAGS.plot:
            update_plot(valid_result_plot)
            tmp_path = os.path.join(storage_path, 'tmpfig_' + str(k_iteration) + '.pdf')
            fig.savefig(tmp_path, format='pdf')

        for key in ['ler', 'loss']:
            results[key + '_list'].append(valid_result[key])
        results['ler_test_list'].append(test_result['ler'])
        results['fr_max_list'].append(firing_rate_stats[1])
        results['fr_avg_list'].append(firing_rate_stats[2])
        idx_min_valid_err = np.argmin(results['ler_list'])
        early_stopping_test_error = results['ler_test_list'][idx_min_valid_err]
        results['early_stopping_test_ler0'] = early_stopping_test_error

        for key, variable in zip(['iteration', 'epoch', 'training_time'],
                                 [k_iteration, dataset.current_epoch, training_time]):
            results[key + '_list'].append(variable)

        with open(os.path.join(storage_path, 'metrics.json'), 'w') as f:
            json.dump(results, f, cls=NumpyAwareEncoder, indent=2)

    t0 = time.time()
    sess.run(train_step, feed_dict=batch_to_feed_dict(dataset.get_next_training_batch(), is_train=True))
    if FLAGS.eprop == 'adaptive':
        sess.run(KolenPollackDecay)
    training_time = time.time() - t0

del sess
