import tensorflow as tf
import numpy as np
import os
import numpy.random as rd
import pickle
import json
import datetime
import errno

def folder_reference(script_name,FLAGS):
    try:
        flag_dict = FLAGS.flag_values_dict()
    except:
        print(
            'Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform lag to dict')
        flag_dict = FLAGS.__flags
    print(json.dumps(flag_dict, indent=4))


    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M__%S_%f")
    time_stamp += str(rd.randint(0, 100))
    folder_path = os.path.join('results',script_name,time_stamp + '_' + FLAGS.comment)
    try:
        os.makedirs(folder_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    class DecimalEncoder(json.JSONEncoder):
        def default(self, o):
            if not isinstance(o, float):
                return float(o)
            return super(DecimalEncoder, self).default(o)

    with open(os.path.join(folder_path,'flags.json'), 'w') as f:
        json.dump(flag_dict, f, cls=DecimalEncoder, indent=4)

    return folder_path

def einsum_bij_jk_to_bik(a,b):
    try:
        n_b = int(a.get_shape()[0])
    except:
        n_b = tf.shape(a)[0]

    try:
        n_i = int(a.get_shape()[1])
    except:
        n_i = tf.shape(a)[1]

    try:
        n_j = int(a.get_shape()[2])
    except:
        n_j = tf.shape(a)[2]

    try:
        n_k = int(b.get_shape()[1])
    except:
        n_k = tf.shape(b)[1]

    a_ = tf.reshape(a,(n_b * n_i,n_j))
    a_b = tf.matmul(a_,b)
    ab = tf.reshape(a_b,(n_b,n_i,n_k))
    return ab

def pad_vector(v, n_time, pad_value=0.):
    if len(v.shape) == 2:
        shp = v.shape
        return np.concatenate([v, pad_value * np.ones((n_time - shp[0], shp[1]))], axis=0)
    elif len(v.shape) == 1:
        shp = v.shape
        return np.concatenate([v, pad_value * np.zeros((n_time - shp[0],))], axis=0)


def sparsity_dense_vector(vector, blank_symbol):
    indices = []
    values = []
    d_vector = np.diff(vector)
    change_indices = np.where(d_vector != 0)[0]
    last_value = blank_symbol
    for ind in change_indices:
        value = vector[ind]
        indices.append(ind)
        values.append(value)
        # last_value=value

    # last_v = blank_symbol
    # for v in values:
    #     assert v != blank_symbol, 'v: {}'.format(blank_symbol)
    #     assert v != last_v, 'v {} last_v {} '.format(v,last_v)
    #     last_v = v

    return np.array(indices, dtype=np.int), np.array(values, dtype=np.int)

def label_stack_to_sparse_tensor(label_stack, blank_symbol):
    sparse_tensor = {'indices': [], 'values': []}

    for i_batch, phns in enumerate(label_stack):
        indices, values = sparsity_dense_vector(phns, blank_symbol)

        sparse_tensor['indices'].append([[i_batch, i_time] for i_time in indices])
        sparse_tensor['values'].append(values)

    sparse_tensor['indices'] = np.concatenate(sparse_tensor['indices'])
    sparse_tensor['values'] = np.concatenate(sparse_tensor['values'])

    return sparse_tensor


class TimitDataset:
    def __init__(self, n_mini_batch, data_path='../datasets/timit_processed', preproc=None,
                 use_reduced_phonem_set=True, return_sparse_phonem_tensor=False,epsilon=1e-10):

        assert preproc is not None
        self.data_path = data_path
        self.preproc = preproc
        self.use_reduced_phonem_set = use_reduced_phonem_set
        self.return_sparse_phn_tensor = return_sparse_phonem_tensor

        self.epsilon = epsilon

        self.n_feats = {
            'fbank': 41 * 3,
            'mfccs': 13 * 3,
            'htk': 13 * 3 if 'htk_mfcc' in data_path else 41 * 3,
            # 'mfccspike': 13 * 31,
            # 'melspec': 16 * 3,
            # 'melspike': 496,
            'cochspec': 86 * 3,
            'cochspike': 86
        }
        self.n_features = self.n_feats[preproc]
        self.n_phns = 39 if use_reduced_phonem_set else 61

        # Load features from files
        self.feature_stack_train, self.phonem_stack_train, self.meta_data_train, _, _ = self.load_data_stack('train')
        self.feature_stack_test, self.phonem_stack_test, self.meta_data_test, self.vocabulary, self.wav_test = \
            self.load_data_stack('test')
        self.feature_stack_develop, self.phonem_stack_develop, self.meta_data_develop, self.vocabulary, self.wav_val = \
            self.load_data_stack('develop')

        def add_derivatives(features):
            n_features = features[0].shape[1]

            # add derivatives:
            get_delta = lambda v : np.concatenate([np.zeros((1, v.shape[1])), v[2:] - v[:-2], np.zeros((1, v.shape[1]))],axis=0)
            d_features = [get_delta(f) for f in features]
            d2_features = [get_delta(f) for f in d_features]

            features = [np.concatenate([f, d_f, d2_f], axis=1) for f,d_f,d2_f in zip(features,d_features,d2_features)]
            assert (features[0].shape[1] == self.n_features)
            return features

        if self.preproc not in ['cochspike', 'htk']:
            self.feature_stack_train = add_derivatives(self.feature_stack_train)
            self.feature_stack_test = add_derivatives(self.feature_stack_test)
            self.feature_stack_develop = add_derivatives(self.feature_stack_develop)

        # normalize the features
        concatenated_training_features = np.concatenate(self.feature_stack_train,axis=0)
        means = np.mean(concatenated_training_features,axis=0)
        stds = np.std(concatenated_training_features,axis=0)

        if self.preproc != 'cochspike':
            self.feature_stack_train = [(f - means) / np.maximum(stds,self.epsilon) for f in self.feature_stack_train]
            self.feature_stack_test = [(f - means) / np.maximum(stds,self.epsilon) for f in self.feature_stack_test]
            self.feature_stack_develop = [(f - means) / np.maximum(stds,self.epsilon) for f in self.feature_stack_develop]

        self.feature_stack_train = np.array(self.feature_stack_train,dtype=object)
        self.feature_stack_test = np.array(self.feature_stack_test,dtype=object)
        self.feature_stack_develop = np.array(self.feature_stack_develop,dtype=object)

        assert (len(self.vocabulary) == self.n_phns)

        self.n_mini_batch = n_mini_batch
        # n_train_total = len(self.feature_stack_train)
        # self.n_validation = int(proportion_validation * n_train_total)+1
        # self.n_train = n_train_total - self.n_validation
        self.n_train = len(self.feature_stack_train)
        self.n_test = len(self.feature_stack_test)
        self.n_develop = len(self.feature_stack_develop)

        print('Dataset sizes: test {} \t train {} \t validation {}'.format(self.n_test,self.n_train,self.n_develop))

        # self.mini_batch_indices,self.validation_indices = self.generate_mini_batch_selection_list()
        self.mini_batch_indices = self.generate_mini_batch_selection_list()
        self.current_epoch = 0
        self.index_current_minibatch = 0

    def reduce_phonem_list(self, phn_list):
        return [self.phonem_reduction_map[k] for k in phn_list]

    def generate_mini_batch_selection_list(self):
        # perm = rd.permutation(self.n_train + self.n_validation)
        # number_of_batches = self.n_train // self.n_mini_batch
        # training_set = perm[:self.n_train]
        # validation_set = perm[self.n_train:]
        perm = rd.permutation(self.n_train)
        number_of_batches = self.n_train // self.n_mini_batch
        return np.array_split(perm, number_of_batches) #,validation_set

    def load_data_stack(self, dataset):
        path = os.path.join(self.data_path, dataset)

        # Define the link to the pickle objects
        if self.preproc == 'fbank':
            feature_path = os.path.join(path, 'filter_banks.pickle')
        elif self.preproc == 'mfccs':
            feature_path = os.path.join(path, 'mfccs.pickle')
        elif self.preproc == 'htk':
            feature_path = os.path.join(path, 'htk.pickle')
        # elif self.preproc == 'mfccspike':
        #     feature_path = os.path.join(path, 'mfcc_spike_stack.pickle')
        # elif self.preproc == 'melspec':
        #     feature_path = os.path.join(path, 'specgram.pickle')
        # elif self.preproc == 'melspike':
        #     feature_path = os.path.join(path, 'spike.pickle')
        elif self.preproc == 'cochspec':
            feature_path = os.path.join(path, 'coch_raw.pickle')
        elif self.preproc == 'cochspike':
            feature_path = os.path.join(path, 'coch_spike.pickle')
        else:
            raise NotImplementedError('Preprocessing %s not available' % self.preproc)

        if self.use_reduced_phonem_set:
            phonem_path = os.path.join(path, 'reduced_phonems.pickle')
            vocabulary_path = os.path.join(path, 'reduced_phonem_list.json')
        else:
            phonem_path = os.path.join(path, 'phonems.pickle')
            vocabulary_path = os.path.join(path, 'phonem_list.json')

        # Load the data
        with open(feature_path, 'rb') as f:
            data_stack = np.array(pickle.load(f))

        with open(phonem_path, 'rb') as f:
            phonem_stack = np.array(pickle.load(f))

            for phns in phonem_stack:
                assert ((np.array(phns) < self.n_phns).all()), 'Found phonems up to {} should be maximum {}'.format(
                    np.max(phns), self.n_phns)

        # Load the vocabulay
        with open(vocabulary_path, 'r') as f:
            vocabulary = json.load(f)

        assert vocabulary[0] == ('sil' if self.use_reduced_phonem_set else 'h#')
        self.silence_symbol_id = 0

        # Load meta data
        with open(os.path.join(path, 'metadata.pickle'), 'rb') as f:
            metadata = pickle.load(f)

        assert vocabulary[0] == ('sil' if self.use_reduced_phonem_set else 'h#')
        self.silence_symbol_id = 0

        with open(os.path.join(path, 'reduced_phn_index_mapping.json'), 'r') as f:
            self.phonem_reduction_map = json.load(f)

        # Load raw audio
        wav_path = os.path.join(path, 'wav.pickle')
        with open(wav_path, 'rb') as f:
            wav_stack = np.array(pickle.load(f))

        return data_stack, phonem_stack, metadata, vocabulary, wav_stack

    def load_features(self, dataset,selection):
        if dataset == 'train':
            feature_stack = self.feature_stack_train[selection]
            phonem_stack = self.phonem_stack_train[selection]
            wavs = None
        elif dataset == 'test':
            feature_stack = self.feature_stack_test[selection]
            phonem_stack = self.phonem_stack_test[selection]
            wavs = self.wav_test[selection]
        elif dataset == 'develop':
            feature_stack = self.feature_stack_develop[selection]
            phonem_stack = self.phonem_stack_develop[selection]
            wavs = self.wav_val[selection]

        seq_len = [feature.shape[0] for feature in feature_stack]

        n_time = np.max([feature.shape[0] for feature in feature_stack])

        features = np.stack([pad_vector(feature, n_time) for feature in feature_stack], axis=0)

        if self.return_sparse_phn_tensor:
            phns = label_stack_to_sparse_tensor(phonem_stack, self.silence_symbol_id)
        else:
            phns = np.stack([pad_vector(phns, n_time, self.silence_symbol_id) for phns in phonem_stack], axis=0)

        return features, phns, seq_len, wavs

    def get_next_training_batch(self):
        features, phns, seq_len, _ = self.load_features('train',
                                                        selection=self.mini_batch_indices[self.index_current_minibatch])

        self.index_current_minibatch += 1
        if self.index_current_minibatch >= len(self.mini_batch_indices):
            self.index_current_minibatch = 0
            self.current_epoch += 1

            #Shuffle the training set after each epoch
            number_of_batches = len(self.mini_batch_indices)
            training_set_indices = np.concatenate(self.mini_batch_indices)
            training_set_indices = rd.permutation(training_set_indices)
            self.mini_batch_indices = np.array_split(training_set_indices, number_of_batches)

        if not self.return_sparse_phn_tensor:
            check = (phns < self.n_phns).all()
        else:
            check = (phns['values'] < self.n_phns).all()

        assert (check), 'Found phonems up to {} should be maximum {}'.format(np.max(phns), self.n_phns)

        return features, phns, seq_len, np.zeros((1, 1))

    def get_test_batch(self):
        return self.load_features('test', np.arange(self.n_test, dtype=np.int))

    def get_next_test_batch(self, selection):
        features, phns, seq_len, wavs = \
            self.load_features('test', selection=selection)

        if not self.return_sparse_phn_tensor:
            check = (phns < self.n_phns).all()
        else:
            check = (phns['values'] < self.n_phns).all()

        assert (check), 'Found phonems up to {} should be maximum {}'.format(np.max(phns), self.n_phns)

        return features, phns, seq_len, wavs

    def get_validation_batch(self):
        return self.load_features('develop', np.arange(self.n_develop, dtype=np.int))

    def get_next_validation_batch(self, selection):
        features, phns, seq_len, wavs = \
            self.load_features('develop', selection=selection)

        if not self.return_sparse_phn_tensor:
            check = (phns < self.n_phns).all()
        else:
            check = (phns['values'] < self.n_phns).all()

        assert (check), 'Found phonems up to {} should be maximum {}'.format(np.max(phns), self.n_phns)

        return features, phns, seq_len, wavs

    def plot_feature(self, ax, feature, phn_vector, text):

        last_time_index = np.where(np.any(feature != 0,axis=1))[0][-1]
        feature = feature[:last_time_index,:]
        phn_vector = phn_vector[:last_time_index]

        ax.imshow(feature.T, origin='lower')
        ax.set_title(text)

        ind_change = np.where(np.diff(phn_vector) != 0)[0]
        phns_change = phn_vector[ind_change]

        ax.set_xticks(np.concatenate([[0], ind_change]))
        tick_labels = [self.vocabulary[k] for k in phns_change]
        tick_labels = ['_' if lab == 'sil' else lab for lab in tick_labels]
        tick_labels.append(' ')
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel('Acoustic feature')
        ax.set_xlabel('Phonems')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
