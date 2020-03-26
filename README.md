Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons  
Authors: G Bellec\*, F Scherr\*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass  
[[Bioarxiv]](https://www.biorxiv.org/content/10.1101/738385v3)  
[[Older pre-print with a related content]](https://arxiv.org/abs/1901.09049)  

The scripts were tested for tensorflow 1.12 or 1.15 and python3.6. More detailed requirements are provided in the different sub-folders.

## Table of contents (e-prop)

- `numerical_verification_eprop_factorization_vs_BPTT.py`: This script verifies numerically the equation (1) of the paper.
It compares numerically the loss gradient computed via BPTT and autodiff against the gradients computed with equation (1), i.e. with eligibility traces and learning signals.

- `numerical_verification_eprop_hardcoded_vs_autodiff.py`: This script verifies numerically that the implementation of e-prop using equation (4) and (25) is strictly equivalent to an alternative implementation using auto-diff.


- `Figure_3_and_S7_e-prop_tutorials`: Code for the temporal credit assignment (Fig. 3) task and the pattern generation task (Fig. S7).
Following section S1.1 from the paper, e-prop can be implemented easily with automatic differentiation.
In the folder, we solve the tasks both with a hardcoded implementation of e-prop through equation (4) and (25) or the autodiff alternative that uses more efficiently the features of tensorflow.

- `Figure_2_TIMIT`: Code to reproduce the results on the phoneme recognition task.

- `Figure_4_and_5_ATARI`: Code for reward based e-prop on ATARI games.  
