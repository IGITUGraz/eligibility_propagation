# Quick start

If you just want to very quick start with the experiment do the following steps:

- Donwload the [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1)
- Process the dataset by running: `python3 timit_processing.py`
- Train an LSTM with symmetric e-prop:  
`PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 solve_timit_with_framewise_lstm.py --eprop=symmetric --preproc=mfccs`


# TIMIT phoneme recognition experiment

There are two versions of this task:

- Framewise classification
- Sequence classification using CTC

## Input preprocessing

Use the `timit_processing.py` script to process the raw [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1)
to features appropriate for training the models.

There are two possibilities for doing this.
Simpler one is using the `python_speech_features` which can be easily installed using `pip`.

A more involved alternative is using the `HTK` software package.
For this you first have to install the [HTK toolkit](http://htk.eng.cam.ac.uk/) on your local machine,
and then process the raw TIMIT dataset using [this fork of asr_processing](https://github.com/dsalaj/asr_preprocessing)
repository. This fork contains the changes which allow running the script on the TIMIT dataset which augments it
by creating the `.htk` files in the dataset.
See [this README](https://github.com/dsalaj/asr_preprocessing/tree/master/timit) for more details.
After augmenting the TIMIT dataset with using asr_processing, simply run the `timit_processing.py` script, which
should in then prepare additional features in the file `htk.pickle`.

Inside the `timit_processing.py` script you should change the `DATA_PATH` variable to point to the source data,
which should be the TIMIT dataset (with or without additional HTK feature files). Start the preprocessing with:

    python3 timit_processing.py


## Framewise training

To train the artificial LSTM network on this task run:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 solve_timit_with_framewise_lstm.py

To train the spiking LSNN network on this task run:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 solve_timit_with_framewise_lsnn.py

> NOTE: running this requres at least a 16GB memory on the GPU. If not available it is possible to train on the CPUs

## Sequence CTC training

To train the artificial LSTM network on this task run:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 solve_timit_with_ctc_in_tensorflow.py

To train the spiking LSNN network on this task run:

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 solve_timit_with_ctc_in_tensorflow.py --model=lsnn

> NOTE: running this requres at least a 24GB memory on the GPU. If not available it is possible to train on the CPUs

## Parameters

By default the above scripts train the models using BPTT. To train with e-prop make use of the `eprop` flag.
For example to train with symmetric e-prop algorithm use: `--eprop=symmetric`
Consult the flags declarations inside the scripts to see the default parameters and ways to configure the experiments.