import numpy as np
from python_speech_features import mfcc,fbank
import os
from sphfile import SPHFile
from scipy.io import wavfile
import csv
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from struct import unpack

DATA_PATH = '../datasets/timit'
TMP_PATH = '/tmp/timit_processing'
OUTPUT_PATH = '../datasets/timit_processed'

interactive_plot_histogram = True
exclude_sa_files = True
normalize_over_training_set = True

n_drs = 8
n_filter_bank = 40
window_step_in_second = 0.01
window_size_in_second = 0.025
SAMPLING_RATE = 16000
window_step_in_samples = int(window_step_in_second*SAMPLING_RATE)+1

dataset_sources = ['train','test', 'test']
dataset_targets = ['train','test','develop']
drs = ['dr'+str(i) for i in range(1,n_drs +1)]

reduced_phn_list = ['sil']
phn_list = ['h#']
word_list = [None]

develop_selection = [
'dr2/mgwt0',
'dr2/mpdf0',
'dr2/mjar0',
'dr2/mmdm2',
'dr2/mmdb1',
'dr4/fgjd0',
'dr4/fjmg0',
'dr4/fadg0',
'dr4/fnmr0',
'dr4/frew0',
'dr4/mteb0',
'dr4/fedw0',
'dr4/mdls0',
'dr4/fsem0',
'dr4/fdms0',
'dr4/mroa0',
'dr4/mbns0',
'dr5/fcal1',
'dr5/fmah0',
'dr5/mrws1',
'dr8/fjsj0',
'dr8/majc0',
'dr6/fdrw0',
'dr6/mjfc0',
'dr6/mrjr0',
'dr7/mdlf0',
'dr7/mers0',
'dr7/fmml0',
'dr7/mrcs0',
'dr7/mrjm4',
'dr7/mdvc0',
'dr1/faks0',
'dr1/fdac1',
'dr1/mjsw0',
'dr1/mreb0',
'dr1/fjem0',
'dr3/mmwh0',
'dr3/mglb0',
'dr3/fkms0',
'dr3/mgjf0',
'dr3/mtdt0',
'dr3/mrtk0',
'dr3/mtaa0',
'dr3/mcsh0',
'dr3/mmjr0',
'dr3/mwjg0',
'dr3/mbdg0',
'dr3/fcmh0',
'dr3/mbwm0',
'dr3/mthc0',]

core_test_selection = [
'dr2/fpas0',
'dr2/mtas1',
'dr2/mwew0',
'dr4/mtls0',
'dr4/mlll0',
'dr4/fjlm0',
'dr5/fnlp0',
'dr5/mbpm0',
'dr5/mklt0',
'dr8/fmld0',
'dr8/mpam0',
'dr8/mjln0',
'dr6/mcmj0',
'dr6/fmgd0',
'dr6/mjdh0',
'dr7/fdhc0',
'dr7/mnjm0',
'dr7/mgrt0',
'dr1/mdab0',
'dr1/felc0',
'dr1/mwbt0',
'dr3/fpkt0',
'dr3/mjmp0',
'dr3/mlnt0',
]

phonem_reduction_table = {
    # according to graves phd
    # https://www.cs.toronto.edu/~graves/phd.pdf
    # 'aa': 'aa',
    'ao': 'aa',

    # 'ah': 'ah',
    'ax': 'ah',
    'ax-h': 'ah',

    # 'er': 'er',
    'axr': 'er',

    # 'hh': 'hh',
    'hv': 'hh',

    # 'ih': 'ih',
    'ix': 'ih',

    # 'l': 'l',
    'el': 'l',

    # 'm': 'm',
    'em': 'm',

    # 'n': 'n',
    'en': 'n',
    'nx': 'n',

    # 'ng': 'ng',
    'eng': 'ng',

    # 'sh': 'sh',
    'zh': 'sh',

    # Silence folding and grouping
    'pcl': 'sil',
    'tcl': 'sil',
    'kcl': 'sil',
    'bcl': 'sil',
    'dcl': 'sil',
    'gcl': 'sil',
    'h#': 'sil',
    'pau': 'sil',
    'epi': 'sil',

    # 'uw': 'uw',
    'ux': 'uw',

    'q': 'sil',
}

def get_phn_or_word_id(phn_or_word,phn_or_word_list,freeze_list):
    inds = np.where(np.array(phn_or_word_list) == phn_or_word)[0]

    assert len(inds) in [0,1]
    if len(inds) == 1:
        return inds[0]

    if freeze_list:
        print('WARNING: List should be frozen (test set) but phn or word \'{}\' is not in list {}.'.format(phn_or_word,phn_or_word_list))

    phn_or_word_list.append(phn_or_word)
    return len(phn_or_word_list) - 1

def get_file_name_tuple_from_speaker_path(speaker_path):
    files = []
    file_names = os.listdir(speaker_path)
    assert(file_names), 'Speaker path: {} is empty ({})'.format(speaker_path,file_names)
    for f in file_names:

        trunc = f[:-4]
        if f[-4:] == '.wav':
            if not(exclude_sa_files) or f[:2] != 'sa':
                wav_file = trunc + '.wav'
                phn_file = trunc + '.phn'
                wrd_file = trunc + '.wrd'
                txt_file = trunc + '.txt'
                files.append((wav_file,phn_file,wrd_file,txt_file))
    return files

def process_wav(path,wav_file):
    assert(wav_file[-3:] == 'wav'), 'Wrong file name, should be a wav: {}'.format(wav_file)
    sphere_file_path = os.path.join(path,wav_file)

    if not(Path(TMP_PATH).exists()):
        os.mkdir(TMP_PATH)

    wav_copy_file_path = os.path.join(TMP_PATH,wav_file) + '_readable'

    sph = SPHFile(sphere_file_path)
    sph.write_wav(wav_copy_file_path)
    FS,wav = wavfile.read(wav_copy_file_path)
    os.remove(wav_copy_file_path)

    mfccs = mfcc(wav, FS, winstep=window_step_in_second, winlen=window_size_in_second)
    fbs,energy = fbank(wav, FS, nfilt=n_filter_bank, winstep=window_step_in_second, winlen=window_size_in_second)
    fbs_with_energy = np.concatenate([fbs,energy[:,None]],axis=1)
    return mfccs,fbs_with_energy,wav,FS

def process_htk(htk_path):

    with open(htk_path, "rb") as f:
        # Read header
        spam = f.read(12)
        frame_num, sampPeriod, sampSize, parmKind = unpack(">IIHH", spam)

        # Read data
        feature_dim = int(sampSize / 4)
        f.seek(12, 0)
        input_data = np.fromfile(f, 'f')
        try:
            input_data = input_data.reshape(-1, feature_dim)
        except:
            print(input_data.shape)
            raise ValueError

        input_data.byteswap(True)

    return input_data, sampPeriod, parmKind

def process_phn_or_word(path, phn_or_word_file, meta_data, reduce_phonem):
    # Generate the phonem vector empty
    vector = np.zeros(meta_data['num_windows'],dtype=int)

    path = os.path.join(path, phn_or_word_file)
    is_phn = True if path[-4:] == '.phn' else False

    # The list of phones are frozen after the pass on the training set
    freeze_ids = meta_data['dataset_source'] in ['test','develop'] and is_phn

    with open(path,'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            row = list(row)
            k_start = int(row[0]) // window_step_in_samples
            k_end = int(row[1]) // window_step_in_samples

            phn_or_word = row[2]

            if is_phn:
                if reduce_phonem:
                    if phn_or_word in phonem_reduction_table.keys():
                        phn_or_word = phonem_reduction_table[phn_or_word]
                    ind = get_phn_or_word_id(phn_or_word,reduced_phn_list,freeze_ids)
                    vector[k_start:k_end] = ind
                    assert(ind < 39),'Found reduced index {} should be lower than 39.'.format(ind)
                else:
                    vector[k_start:k_end] = get_phn_or_word_id(phn_or_word,phn_list,freeze_ids)
            else:
                vector[k_start:k_end] = get_phn_or_word_id(phn_or_word,word_list,freeze_ids)
    return vector

def process_txt(path,txt_file):
    txt_path = os.path.join(path,txt_file)

    with open(txt_path,'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            row = list(row)
            start_time = int(row[0])
            end_time = int(row[1])
            sentence = ' '.join(row[2:])
            break

    return start_time,end_time,sentence

if __name__ == '__main__':
    if interactive_plot_histogram:
        plt.ion()
        fig,axes = plt.subplots(3, figsize=(5, 6))
        ax = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]

    for dataset_source,dataset_target in zip(dataset_sources,dataset_targets):
        print('Processing {} dialects for dataset: '.format(n_drs) + dataset_target)

        metadata_stack = []
        wav_stack = []
        htk_stack = []
        mfcc_stack = []
        fbank_stack = []
        phn_stack = []
        reduced_phn_stack = []
        word_stack = []
        dataset_file_count = 0

        for dr in drs:
            dialect_file_count = 0
            dr_path = os.path.join(DATA_PATH,dataset_source,dr)
            full_speaker_path_list = os.listdir(dr_path)
            selected_speaker_path_list = []
            for speaker in full_speaker_path_list:

                speaker_reference = dr + '/' + speaker

                use_speaker = False
                if dataset_source == 'train': use_speaker=True
                if speaker_reference in develop_selection and dataset_target == 'develop': use_speaker=True
                if speaker_reference in core_test_selection and dataset_target == 'test': use_speaker=True


                if use_speaker:
                    selected_speaker_path_list.append(speaker)

            for speaker in selected_speaker_path_list:

                speaker_path = os.path.join(dr_path,speaker)

                for wav_file,phn_file,wrd_file,txt_file in get_file_name_tuple_from_speaker_path(speaker_path):
                    dialect_file_count += 1
                    # Process txt file
                    sample_start, sample_end, text = process_txt(speaker_path, txt_file)

                    # Process wave file
                    mfccs,fbanks,wav,FS = process_wav(speaker_path,wav_file)

                    htk_path = os.path.join(speaker_path, wav_file[:-3] + 'htk')
                    if os.path.isfile(htk_path):  # prepare HTK features only if available
                        htk_feats, _, _ = process_htk(htk_path)
                        htk_feats = np.pad(htk_feats, ((0, 1), (0, 0)), mode='edge')
                        assert mfccs.shape[0] == htk_feats.shape[0]
                        htk_stack.append(htk_feats)

                    duration = (sample_end - sample_start) / SAMPLING_RATE
                    meta_data = {'dr': dr,
                                 'speaker': speaker,
                                 'sample_start': sample_start,
                                 'sample_end': sample_end,
                                 'text': text,
                                 'dataset_source': dataset_source,
                                 'dataset_target': dataset_target,
                                 'duration': duration,
                                 'num_windows': mfccs.shape[0]
                                }

                    # Process phn and words
                    phns = process_phn_or_word(speaker_path,phn_file,meta_data,False)
                    reduced_phns = process_phn_or_word(speaker_path,phn_file,meta_data,True)
                    words = process_phn_or_word(speaker_path,wrd_file,meta_data,False)

                    assert(len(phns) == mfccs.shape[0]), 'Mismatch of sizes, phns: {} and mfccs {}'.format(np.shape(phns),np.shape(mfccs))

                    assert( (np.array(reduced_phns) < 39).all() )
                    assert( (np.array(phns) < 123).all() )

                    metadata_stack.append(meta_data)
                    wav_stack.append(wav)
                    mfcc_stack.append(mfccs)
                    fbank_stack.append(fbanks)
                    phn_stack.append(phns)
                    reduced_phn_stack.append(reduced_phns)
                    word_stack.append(words)

            print('Dialect {}: {} speakers - {} audio/label file pairs'.format(dr,len(selected_speaker_path_list),dialect_file_count))
            dataset_file_count += dialect_file_count

            if interactive_plot_histogram and dataset_file_count > 0:
                tfrom = 5
                tto = 60
                ax.clear()
                i = len(metadata_stack)-1
                plot_features = np.log(fbank_stack[i][:, :-1]**2).T
                plot_features = plot_features[:, tfrom:tto]
                ax.imshow(plot_features, origin='lower', aspect="auto")
                # im = ax.imshow(htk_stack[i][:, :40].T, origin='lower')
                ax.set_title(metadata_stack[i]['text'])
                ind_change = np.where(np.diff(reduced_phn_stack[i]) !=0)[0]
                phns_change = reduced_phn_stack[i][ind_change]
                phone_ticks = np.concatenate([[0],ind_change])
                trimmed_indices = np.where(np.logical_and(phone_ticks >= tfrom, phone_ticks < tto))
                ax.set_xticks(phone_ticks[trimmed_indices])
                tick_labels = [reduced_phn_list[k] for k in phns_change[trimmed_indices]]
                tick_labels = ['_' if lab == 'sil' else lab for lab in tick_labels]
                tick_labels.append(' ')
                # ax.set_xlim(tfrom, tto)
                ax.set_yticks([])
                ax.set_xticklabels(tick_labels)
                ax.set_ylabel('filter bank')
                ax.set_xlabel('phones')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax2.clear()
                wav_from = tfrom * window_step_in_second
                wav_to = tto * window_step_in_second
                signal = wav_stack[i]
                time = np.linspace(0, len(signal) / SAMPLING_RATE, num=len(signal))
                ax2.plot(time, signal)
                ax2.set_xlim([time[0], time[-1]])
                ax2.set_xlim([wav_from, wav_to])
                ax2.set_yticks([])
                ax2.set_xlabel('seconds')
                ax2.set_ylabel('audio')
                ax2.spines['right'].set_visible(False)
                ax2.spines['top'].set_visible(False)

                ax3.clear()
                fs = 8
                phm = reduced_phn_stack[i][tfrom:tto]
                corr_phm = np.expand_dims(np.ones_like(phm), axis=1)
                ax3.matshow(corr_phm, cmap='Greys', alpha=0.5, aspect='auto')
                diffps = []
                for j in range(phm.shape[0]):
                    c = phm[j]
                    phone = reduced_phn_list[c]
                    phone = '_' if phone == 'sil' else phone
                    if len(diffps) > 0 and phone != diffps[-1]:
                        diffps.append(phone)
                    if len(diffps) == 0:
                        diffps.append(phone)

                    ax3.text(j, 0, phone, va='center', ha='center', fontsize=fs, rotation=90)
                ax3.set_xlim(-0.5, phm.shape[0] - 0.5)
                ax3.set_ylim(-0.5, 0.5)
                ax3.set_xticks([i - 0.5 for i in range(phm.shape[0])], minor=True)
                ax3.set_xticklabels([], minor=True)
                # ax3.set_yticks([-0.5, 0.5], minor=True)
                # ax3.set_yticklabels([], minor=True)
                ax3.set_yticks([])
                ax3.set_yticklabels([], fontsize=fs)
                ax3.grid(which='minor', linestyle='-', color='black')
                ax3.spines['right'].set_visible(True)
                ax3.spines['bottom'].set_visible(True)
                for tic in ax3.xaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                    tic.label1On = tic.label2On = False
                for tic in ax3.xaxis.get_minor_ticks():
                    tic.tick1On = tic.tick2On = False
                    tic.label1On = tic.label2On = False
                ax3.set_ylabel('framewise target')

                plt.tight_layout()
                plt.draw()
                if (dr == "dr3"):
                    plt.savefig("timit_input.svg", format='svg', dpi=600)
                plt.pause(1)

        print('Processed {} audio/label file pairs for dataset {} '.format(dataset_file_count,dataset_target))

        if normalize_over_training_set:

            if dataset_target == 'train':
                concatenated_mfccs = np.concatenate(mfcc_stack,axis=0)
                concatenated_fbanks = np.concatenate(fbank_stack,axis=0)

                mfcc_means = concatenated_mfccs.mean(axis=0)
                mfcc_stds = concatenated_mfccs.std(axis=0)

                fbank_means = concatenated_fbanks.mean(axis=0)
                fbank_stds = concatenated_fbanks.std(axis=0)

                if len(htk_stack) > 0:
                    concatenated_htks = np.concatenate(htk_stack, axis=0)
                    htk_means = concatenated_htks.mean(axis=0)
                    htk_stds = concatenated_htks.std(axis=0)

            mfcc_stack = [(mfccs - mfcc_means) / mfcc_stds for mfccs in mfcc_stack]
            fbank_stack = [(fbanks - fbank_means) / fbank_stds for fbanks in fbank_stack]
            if len(htk_stack) > 0:
                htk_stack = [(feat - htk_means) / htk_stds for feat in htk_stack]

        assert(len(reduced_phn_list) == 39),'Encountered {} phonems in the dataset, 39 are expected in timit.'.format(
            len(reduced_phn_list))
        assert(len(phn_list) == 61),'Encountered {} phonems in the dataset, 64 are expected in timit.'.format(
            len(phn_list))

        print('Writing the processed data in pickle and json files.')
        if not(Path(OUTPUT_PATH).is_dir()):
            os.mkdir(OUTPUT_PATH)

        output_dataset_path = os.path.join(OUTPUT_PATH, dataset_target)
        if not(Path(output_dataset_path).is_dir()):
            os.mkdir(output_dataset_path)

        for stack,file_name in zip(
                [metadata_stack,wav_stack,mfcc_stack,fbank_stack,phn_stack,
                 reduced_phn_stack,word_stack,htk_stack],
                ['metadata.pickle','wav.pickle','mfccs.pickle','filter_banks.pickle','phonems.pickle',
                 'reduced_phonems.pickle','words.pickle','htk.pickle']):
            with open(os.path.join(output_dataset_path, file_name), 'wb') as f:
                pickle.dump(stack,f,protocol=4)


        def find_reduced_phn_id(phn):
            if phn in phonem_reduction_table.keys():
                return reduced_phn_list.index(phonem_reduction_table[phn])
            else:
                return reduced_phn_list.index(phn)

        reduced_phn_index_mapping = [find_reduced_phn_id(phn) for phn in phn_list]

        for label_list,file_name in zip([word_list,phn_list,reduced_phn_list,reduced_phn_index_mapping],
                                        ['word_list.json','phonem_list.json','reduced_phonem_list.json','reduced_phn_index_mapping.json']):
            with open(os.path.join(output_dataset_path, file_name), 'w') as f:
                json.dump(label_list,f,indent=4)

