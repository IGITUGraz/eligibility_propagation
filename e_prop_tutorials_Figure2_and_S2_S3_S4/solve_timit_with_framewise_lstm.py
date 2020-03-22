import datetime
import tensorflow as tf
import numpy as np
import numpy.random as rd
import time
import os
import json
import errno
from tools import TimitDataset, einsum_bij_jk_to_bik, NpEncoder, pad_vector
from lstm_eprop_model import CustomLSTM

FLAGS = tf.app.flags.FLAGS


def flag_to_dict(FLAG):
    if float(tf.__version__[2:]) >= 5:
        flag_dict = FLAG.flag_values_dict()
    else:
        flag_dict = FLAG.__flags
    return flag_dict


# Accessible parameter from the shell
tf.app.flags.DEFINE_string('comment', '', 'comment attached to output filenames')
tf.app.flags.DEFINE_string('dataset', '../datasets/timit_processed', 'Path to dataset to use')
tf.app.flags.DEFINE_string('preproc', 'htk', 'Input preprocessing: fbank, mfccs, cochspec, cochspike, htk')
tf.app.flags.DEFINE_string('eprop', None, 'options: [None, symmetric, adaptive, random], None means use BPTT')
tf.app.flags.DEFINE_bool('adam', True, 'use ADAM instead of standard SGD')
tf.app.flags.DEFINE_bool('lstm', True, 'plot regularly the predicitons')
#
tf.app.flags.DEFINE_integer('seed', -1, 'seed number')
tf.app.flags.DEFINE_integer('n_epochs', 30, 'number of iteration ')
tf.app.flags.DEFINE_integer('n_lstm', 200, 'number of lstm cells')
tf.app.flags.DEFINE_integer('print_every', 100, 'print every and store accuracy')
tf.app.flags.DEFINE_integer('lr_decay_every', 500, 'Decay every')
tf.app.flags.DEFINE_integer('batch', 32, 'mini_batch size')
#
tf.app.flags.DEFINE_float('init_scale', 0., 'Provide the scaling of the weights at initialization')
tf.app.flags.DEFINE_float('l2', 1e-5, 'l2 regularization')
tf.app.flags.DEFINE_float('lr_decay', .3, 'Learning rate decay')
tf.app.flags.DEFINE_float('lr_init', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('adam_epsilon', 1e-5, 'Epsilon parameter in adam to cut gradients with small variance')
tf.app.flags.DEFINE_float('readout_decay', 1e-3, 'weight decay of readout and broadcast weights 0.001')

#
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M__%S_%f")
script_name = os.path.basename(__file__)[:-3]
filename = time_stamp+'_'+FLAGS.comment
result_doc_name = os.path.join('results', script_name, filename)

# After processing the data, this object loads it and prepare it.
dataset = TimitDataset(FLAGS.batch, data_path=FLAGS.dataset, preproc=FLAGS.preproc)

# Placeholders loaded from data
features = tf.placeholder(shape=(None, None, dataset.n_features), dtype=tf.float32, name='Features')
phns = tf.placeholder(shape=(None, None), dtype=tf.int64, name='Labels')
weighted_relevant_mask = tf.placeholder(shape=(None, None), dtype=tf.float32)
audio = tf.placeholder(shape=(None, None), dtype=tf.float32, name='Audio')

# Non-trainable variables that are used to implement a decaying learning rate and count the iterations
lr = tf.Variable(FLAGS.lr_init, dtype=tf.float32, trainable=False)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
lr_update = tf.assign(lr, lr * FLAGS.lr_decay)


def batch_to_feed_dict(batch):
    '''
    Create the dictionnary that is fed into the Session.run(..) calls.
    :param batch:
    :return:
    '''
    features_np, phns_np, seq_len_np, wav_np = batch
    # print("wav_np", wav_np.shape)
    n_time = max([len(i) for i in wav_np])
    wav_np = np.stack([pad_vector(w, n_time) for w in wav_np], axis=0)
    # print("wav_np", wav_np.shape)
    n_batch, n_time = phns_np.shape
    relevance_mask_np = [(np.arange(n_time) < seq_len_np[i]) / seq_len_np[i] for i in range(n_batch)]

    return {features: features_np, phns: phns_np, weighted_relevant_mask: relevance_mask_np, audio: wav_np}


# Cell model used to solve the task, we have two because we used a bi-directional LSTM cell
if FLAGS.lstm:
    cell_forward = CustomLSTM(FLAGS.n_lstm, stop_gradients=FLAGS.eprop is not None)
    cell_backward = CustomLSTM(FLAGS.n_lstm, stop_gradients=FLAGS.eprop is not None)
else:
    cell_forward = tf.contrib.rnn.BasicRNNCell(FLAGS.n_lstm)
    cell_backward = tf.contrib.rnn.BasicRNNCell(FLAGS.n_lstm)

print("Using " + type(cell_forward).__name__)
result_file_name = result_doc_name + type(cell_forward).__name__ + '_' + time_stamp + '_' + FLAGS.comment
try:
    os.makedirs(result_file_name)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
print("SAVING RESULT TO:", result_file_name)
flagdict = flag_to_dict(FLAGS)
with open(os.path.join(result_file_name, 'flags.json'), 'w') as f:
    json.dump(flagdict, f, indent=2)

# In case we want to control how the LSTM weights are initialized.
if FLAGS.init_scale > 0:
    initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
else:
    initializer = None

# Define the graph for the RNNs processing
with tf.variable_scope('RNNs', initializer=initializer):
    outputs_forward, _ = tf.nn.dynamic_rnn(cell_forward, features, dtype=tf.float32, scope='ForwardRNN')
    outputs_backward, _ = tf.nn.dynamic_rnn(cell_backward, tf.reverse(features, axis=[1]), dtype=tf.float32,
                                            scope='BackwardRNN')
    outputs_backward = tf.reverse(outputs_backward, axis=[1])
    outputs = tf.concat([outputs_forward, outputs_backward], axis=2)


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
    w_out_init = rd.randn(FLAGS.n_lstm * 2, dataset.n_phns) / np.sqrt(FLAGS.n_lstm * 2)  # original
    w_out = tf.Variable(w_out_init, dtype=tf.float32)
    if FLAGS.eprop in ['random']:
        BA_out = tf.constant(rd.randn(FLAGS.n_lstm * 2, dataset.n_phns),
                             dtype=tf.float32, name='BroadcastWeights')

        BA_out = tf.get_variable(name="BAout", initializer=BA_out, dtype=tf.float32)
        phn_logits = BA_logits(outputs, w_out, BA_out)
    elif FLAGS.eprop in ['adaptive']:
        BA_out = tf.constant(rd.randn(FLAGS.n_lstm * 2, dataset.n_phns) / np.sqrt(FLAGS.n_lstm * 2),
                             dtype=tf.float32, name='BroadcastWeights')

        BA_out = tf.get_variable(name="BAout", initializer=BA_out, dtype=tf.float32)
        phn_logits = BA_logits(outputs, w_out, BA_out)
    else:
        phn_logits = einsum_bij_jk_to_bik(outputs, w_out)
    b_out = tf.Variable(np.zeros(dataset.n_phns), dtype=tf.float32)
    phn_logits = phn_logits + b_out

if FLAGS.eprop == 'adaptive':
    weight_decay = tf.constant(FLAGS.readout_decay, dtype=tf.float32)
    w_out_decay = tf.assign(w_out, w_out - weight_decay * w_out)
    BA_decay = tf.assign(BA_out, BA_out - weight_decay * BA_out)
    KolenPollackDecay = [BA_decay, w_out_decay]

# Capute are weight variable in a single list
weight_list = [cell_forward.trainable_weights[0]] + [cell_backward.trainable_weights[0]] + [w_out]

# Define the graph for the loss function and the definition of the error
with tf.name_scope('Loss'):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=phns, logits=phn_logits)
    loss = tf.reduce_mean(loss)
    if FLAGS.l2 > 0:
        losses_l2 = [tf.reduce_sum(tf.square(w)) for w in tf.trainable_variables()]
        loss += FLAGS.l2 * tf.reduce_sum(losses_l2)

    phn_prediction = tf.argmax(phn_logits, axis=2)
    is_correct = tf.equal(phns, phn_prediction)
    is_correct_float = tf.cast(is_correct, dtype=tf.float32)
    ler = tf.reduce_sum(is_correct_float * weighted_relevant_mask, axis=1)
    ler = 1. - tf.reduce_mean(ler)

# Define the training step operation
with tf.name_scope('Train'):
    if not FLAGS.adam:
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    else:
        train_step = tf.train.AdamOptimizer(lr, epsilon=FLAGS.adam_epsilon).minimize(loss, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

results = {
    'loss_list': [],
    'ler_list': [],
    'valid_loss_list': [],
    'valid_ler_list': [],
    'n_synapse': [],
    'iteration_list': [],
    'epoch_list': [],
    'training_time_list': []}

weights = {
    'w_fw': cell_forward._kernel,
    'w_bw': cell_backward._kernel,
    'b_fw': cell_forward._bias,
    'b_bw': cell_backward._bias,
    'w_out': w_out,
    'b_out': b_out,
}

result_plot_tensors = {
    'features': features,
    'phns': phns,
    'phn_predictions': phn_prediction,
    'outputs': outputs,
    'phn_logits': phn_logits,
    'audio': audio,
}

training_time = 0
testing_time = 0
test_result_tensors = {'ler': ler, 'loss': loss}
train_result_tensors = {'train_step': train_step}
test_result = sess.run([ler, loss], feed_dict=batch_to_feed_dict(dataset.get_test_batch()))
valid_result = sess.run([ler, loss], feed_dict=batch_to_feed_dict(dataset.get_validation_batch()))

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print("TOTAL NUM OF PARAMETERS = ", total_parameters)

min_valid_err = 1.
while dataset.current_epoch <= FLAGS.n_epochs:
    k_iteration = sess.run(global_step)
    if np.mod(k_iteration, FLAGS.lr_decay_every) == 0 and k_iteration > 0:
        sess.run(lr_update)
        print('Decay learning rate: {:.2g}'.format(sess.run(lr)))

    if np.mod(k_iteration, FLAGS.print_every) == 0:
        t0 = time.time()
        test_result, weight_results = sess.run([test_result_tensors, weights],
                                               feed_dict=batch_to_feed_dict(dataset.get_test_batch()))
        valid_result = sess.run(test_result_tensors, feed_dict=batch_to_feed_dict(dataset.get_validation_batch()))
        testing_time = time.time() - t0

        print('Epoch: {} \t time/it: {:.2f} s \t time/test: {:.2f} \t it: {} \t '
              'PER: {:.3g} (valid) \t loss {:.3g} (valid) \t PER: {:.3g} (test)'
              .format(dataset.current_epoch, training_time, testing_time, k_iteration, valid_result['ler'],
                      valid_result['loss'], test_result['ler']))

        for key, value in test_result.items():
            results[key + '_list'].append(value)

        results['valid_loss_list'].append(valid_result['loss'])
        results['valid_ler_list'].append(valid_result['ler'])

        # Saving parameters
        if valid_result['ler'] < min_valid_err:
            min_valid_err = valid_result['ler']
            saver.save(sess, os.path.join(result_file_name, 'model.ckpt'))
            saver.export_meta_graph(os.path.join(result_file_name, 'graph.meta'))
            np.save(os.path.join(result_file_name, 'weights.npy'), weight_results)
        idx_min_valid_err = np.argmin(results['valid_ler_list'])
        early_stopping_test_error = results['ler_list'][idx_min_valid_err]
        results['early_stopping_test_ler'] = early_stopping_test_error
        with open(os.path.join(result_file_name, 'metrics.json'), 'w') as f:
            json.dump(results, f, cls=NpEncoder, indent=4)

    t0 = time.time()
    sess.run(train_step, feed_dict=batch_to_feed_dict(dataset.get_next_training_batch()))
    if FLAGS.eprop == 'adaptive':
        sess.run(KolenPollackDecay)
    training_time = time.time() - t0

    for key, variable in zip(['iteration', 'epoch', 'training_time'],
                             [k_iteration, dataset.current_epoch, training_time]):
        results[key + '_list'].append(variable)

del sess
