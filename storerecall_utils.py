import numpy as np
import numpy.random as rd
import tensorflow as tf


# Variations of sequence with specific delay for plotting
def gen_custom_delay_batch(seq_len, seq_delay, batch_size):
    assert type(seq_delay) is int
    assert 2 + 1 + seq_delay + 1 < seq_len

    def gen_custom_delay_input(seq_len, seq_delay):
        seq_delay = 1 + np.random.choice(seq_len - 2) if seq_delay == 0 else seq_delay
        return [np.random.choice([0, 1]) for _ in range(2)] + \
               [2] + [np.random.choice([0, 1]) for _ in range(seq_delay)] + [3] + \
               [np.random.choice([0, 1]) for _ in range(seq_len - (seq_delay + 4))]

    return np.array([gen_custom_delay_input(seq_len, seq_delay) for i in range(batch_size)])


def error_rate(z, num_Y, num_X, n_character):
    # Find the recall index
    n_recall_symbol = n_character + 1
    shp = tf.shape(num_X)

    # Translate the one hot into ints
    shp_x = tf.shape(num_Y)
    z = tf.reduce_mean(tf.reshape(tf.sign(z), (shp_x[0], shp_x[1], -1, 2)), 2)
    char_predict = tf.argmax(z, axis=2)
    char_true = num_Y
    char_input = num_X

    # error rate 1) Wrong characters
    char_correct = tf.cast(tf.equal(char_predict, char_true), tf.float32)
    character_errors = tf.reduce_mean(1 - char_correct)

    # error rate 2) wrong recalls
    recall_mask = tf.equal(char_input, n_recall_symbol)
    recalls_predict = tf.boolean_mask(char_predict, recall_mask)
    recalls_true = tf.boolean_mask(char_true, recall_mask)

    recall_correct = tf.equal(recalls_predict, recalls_true)
    recall_errors = tf.reduce_mean(tf.cast(tf.logical_not(recall_correct), tf.float32))

    # Get wrong samples
    sentence_id = tf.tile(tf.expand_dims(tf.range(shp[0]), axis=1), (1, shp[1]))
    recall_sentence_id = tf.boolean_mask(sentence_id, recall_mask)
    false_sentence_id_list = tf.boolean_mask(recall_sentence_id, tf.logical_not(recall_correct))

    return character_errors, recall_errors, false_sentence_id_list


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern,list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes


def validity_test(seq, recall_char, store_char):
    is_valid = True

    # At least a store, a digit and a recall
    if np.max(seq == recall_char) == 0 or np.max(seq == store_char) == 0 or np.max(seq < store_char) == 0:
        is_valid = False

    # First store before first recall
    t_first_recall = np.argmax(seq == recall_char)
    t_first_store = np.argmax(seq == store_char)
    if t_first_recall < t_first_store:
        is_valid = False

    # Last recall after last store
    t_last_recall = - np.argmax(seq[::-1] == recall_char)
    t_last_store = - np.argmax(seq[::-1] == store_char)
    if t_last_recall < t_last_store:
        is_valid = False

    # Always a digit after a store
    t_store_list = np.where(seq == store_char)[0]
    for t_store in t_store_list:
        if t_store == seq.size - 1 or seq[t_store + 1] in [recall_char, store_char]:
            is_valid = False
            break

    # Between two recall there is a store
    t_recall_list = np.where(seq == recall_char)[0]
    for k, t_recall in enumerate(t_recall_list[:-1]):
        next_t_recall = t_recall_list[k + 1]

        is_store_between = np.logical_and(t_recall < t_store_list, t_store_list < next_t_recall)
        if not (is_store_between.any()):
            is_valid = False

    # Between two store there is a recall
    for k, t_store in enumerate(t_store_list[:-1]):
        next_t_store = t_store_list[k + 1]

        is_recall_between = np.logical_and(t_store < t_recall_list, t_recall_list < next_t_store)
        if not (is_recall_between.any()):
            is_valid = False
    return is_valid


def generate_input_with_prob(batch_size, length, recall_char, store_char, prob_bit_to_store,
                             prob_bit_to_recall):
    input_nums = np.zeros((batch_size, length), dtype=int)

    for b in range(batch_size):
        last_signal = recall_char

        # init a sequence
        is_valid = False
        seq = rd.choice([0, 1], size=length)

        while not is_valid:
            seq = rd.choice([0, 1], size=length)
            for t in range(length):
                # If the last symbol is a recall we wait for a store
                if last_signal == recall_char and rd.rand() < prob_bit_to_store:
                    seq[t] = store_char
                    last_signal = store_char

                # Otherwise we wait for a recall
                elif last_signal == store_char and rd.rand() < prob_bit_to_recall:
                    seq[t] = recall_char
                    last_signal = recall_char

            is_valid = validity_test(seq, recall_char, store_char)

        input_nums[b, :] = seq

    return input_nums


def generate_data(batch_size, length, n_character, prob_bit_to_store=1. / 3, prob_bit_to_recall=1. / 5, input_nums=None,
                  with_prob=True, delay=None):

    store_char = n_character
    recall_char = n_character + 1

    # Generate the input data
    if input_nums is None:
        if with_prob and prob_bit_to_store < 1. and prob_bit_to_recall < 1.:
            input_nums = generate_input_with_prob(batch_size, length, recall_char, store_char,
                                                  prob_bit_to_store, prob_bit_to_recall)
        else:
            raise ValueError("Only use input generated with probabilities")

    input_nums = np.array(input_nums)

    # generate the output
    target_nums = input_nums.copy()
    inds_recall = np.where(input_nums == recall_char)
    for k_trial, k_t in zip(inds_recall[0], inds_recall[1]):
        assert k_t > 0, 'A recall is put at the beginning to avoid this'
        store_list = np.where(input_nums[k_trial, :k_t] == store_char)[0]
        previous_store_t = store_list[-1]
        target_nums[k_trial, k_t] = input_nums[k_trial, previous_store_t + 1]

    memory_nums = np.ones_like(input_nums) * store_char
    for k_trial in range(batch_size):
        t_store_list = np.where(input_nums[k_trial, :] == store_char)[0]
        for t_store in np.sort(t_store_list):
            if t_store < length - 1:
                memory_nums[k_trial, t_store:] = input_nums[k_trial, t_store + 1]

    return input_nums, target_nums, memory_nums


def generate_mikolov_data(batch_size, length, n_character, with_prob, prob_bit_to_recall,
                          prob_bit_to_store, override_input=None, delay=None):
    if n_character > 2:
        raise NotImplementedError("Not implemented for n_character != 2")
    total_character = n_character + 2
    recall_character = total_character - 1
    store_character = recall_character - 1
    i_1 = np.zeros((batch_size, length), dtype=float)
    i_2 = np.zeros((batch_size, length), dtype=float)
    store = np.zeros((batch_size, length), dtype=float)
    recall = np.zeros((batch_size, length), dtype=float)
    channels = [i_1, i_2, store, recall]
    input_nums, target_nums, memory_nums = generate_data(batch_size, length, n_character,
                                                         with_prob=with_prob, prob_bit_to_recall=prob_bit_to_recall,
                                                         prob_bit_to_store=prob_bit_to_store, input_nums=override_input,
                                                         delay=delay)
    for c in range(total_character):
        channels[c] = np.isin(input_nums, [c]).astype(int)

    for b in range(batch_size):
        for i in range(length):
            if channels[store_character][b,i] == 1:
                # copy next input to concurrent step with store
                channels[0][b,i] = channels[0][b,i+1]
                channels[1][b,i] = channels[1][b,i+1]
                # sometimes inverse the next input
                if rd.uniform() < 0.5 and override_input is None:
                    channels[0][b, i+1] = 1 - channels[0][b, i + 1]
                    channels[1][b, i+1] = 1 - channels[1][b, i + 1]
    return channels, target_nums, memory_nums, input_nums


def generate_permutation_arr(batch_size, n_in):
    idx_perm = np.zeros((batch_size, n_in), dtype=np.int)
    idx = np.arange(n_in)
    for b in range(batch_size):
        idx_perm[b, :] = np.random.permutation(idx)
    return idx_perm


def generate_storerecall_data(batch_size, sentence_length, n_character, n_charac_duration, n_neuron, f0=200 / 1000,
                                 with_prob=True, prob_signals=1 / 5, override_input=None, delay=None, shuffle_input=None):

    channels, target_nums, memory_nums, input_nums = generate_mikolov_data(
        batch_size, sentence_length, n_character, with_prob=with_prob, prob_bit_to_recall=prob_signals,
        prob_bit_to_store=prob_signals, override_input=override_input, delay=delay)

    total_character = n_character + 2  # number of input gates
    recall_character = total_character - 1
    store_character = recall_character - 1

    neuron_split = np.array_split(np.arange(n_neuron), total_character)
    lstm_in_rates = np.zeros((batch_size, sentence_length*n_charac_duration, n_neuron))
    in_gates = channels

    for b in range(batch_size):
        for c in range(sentence_length):
            for in_gate_i, n_group in enumerate(neuron_split):
                lstm_in_rates[b, c*n_charac_duration:(c+1)*n_charac_duration, n_group] = in_gates[in_gate_i][b][c] * f0

    if shuffle_input is not None:
        if np.isscalar(shuffle_input):
            idx_perm = generate_permutation_arr(batch_size, int(n_neuron / n_character))
        else:
            idx_perm = shuffle_input
        lstm_in_rates_per = lstm_in_rates
        for b in range(batch_size):
            for i in range(sentence_length*n_charac_duration):
                lstm_in_rates_per[b][i][:int(n_neuron / n_character)] = lstm_in_rates[b][i][idx_perm[b, :]]
        spikes = generate_poisson_noise_np(lstm_in_rates_per)

    else:
        spikes = generate_poisson_noise_np(lstm_in_rates)

    target_sequence = np.repeat(target_nums, repeats=n_charac_duration, axis=1)
    # Generate the recall mask
    is_recall_table = np.zeros((total_character, n_charac_duration), dtype=bool)
    is_recall_table[recall_character, :] = True
    is_recall = np.concatenate([is_recall_table[input_nums][:, k] for k in range(sentence_length)], axis=1)

    return spikes, is_recall, target_sequence, None, input_nums, target_nums


