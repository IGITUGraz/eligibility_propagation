## E-prop: the code repository

This is the official `e-prop` repository provided by the authors of the e-prop paper [1].
This repository is split in three sub-folders, and each of them provides the code to reproduce different figures of the paper and contains its own `README.md` with more specific information.

Two additional scripts `numerical_verification_...py` are providing numerical verifications that are relevant to the papers.
More information are provided in the script headers.

[1] A solution to the learning dilemma for recurrent networks of spiking neurons  
G Bellec\*, F Scherr\*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass  
[[Bioarxiv]](https://www.biorxiv.org/content/10.1101/738385v3)  
[[Older pre-print with a related content]](https://arxiv.org/abs/1901.09049)  


## Overview

- `numerical_verification_eprop_factorization_vs_BPTT.py`: This script verifies numerically the equation (1) of the paper.
It compares numerically the loss gradient computed via BPTT and autodiff against the gradients computed with equation (1), i.e. with eligibility traces and learning signals.
Installation requirements for this script can be found in `Figure_3_and_S7_e-prop_tutorials_`.

- `numerical_verification_eprop_hardcoded_vs_autodiff.py`:
This script verifies numerically that the implementation of e-prop using equation (4) and (25) is strictly equivalent to an alternative implementation using auto-diff as explained in section S1.1 of the paper.
It justifies why we can use equivalently either or both of these implementations in each simulation.
Installation requirements for this script can be found in `Figure_3_and_S7_e-prop_tutorials_`.


- `Figure_3_and_S7_e-prop_tutorials`: Code for the temporal credit assignment (Fig. 3) task and the pattern generation task (Fig. S7).
In the folder, we solve the tasks both with a hardcoded implementation of e-prop (i.e. equations (4) and (25)) and with the autodiff alternative.

- `Figure_2_TIMIT`: Code to reproduce the results on the phoneme recognition task.

- `Figure_4_and_5_ATARI`: Code for reward based e-prop on ATARI games.  


## Installation guide and System requirements

The scripts were tested for tensorflow 1.14 to 1.15 and python3.6 or 3.7 depending on the simulations (see sub-folders for details).
The detailed list of package requirements are provided in the file `requirements.txt` of each different sub-folder.
We recommend using linux and conda 4.8.2 to install those packages.

For instance to reproduce Figure 3, install conda and proceed as follows:

```
cd Figure_3_and_S7_e-prop_tutorials
conda create --name eprop-figure3 python==3.6.2
conda activate eprop-figure3
conda install --file requirements.txt
python tutorial_evidence_accumulation_with_alif.py
```

Warning: After installation, if your computer is equipped with a GPU, you might want to reinstall
the gpu-compatible version of tensorflow.