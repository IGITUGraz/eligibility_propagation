Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons  
Authors: G Bellec\*, F Scherr\*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass  
[[Bioarxiv]](https://www.biorxiv.org/content/10.1101/738385v3)  
[[Older pre-print with a related content]](https://arxiv.org/abs/1901.09049)  

## Table of contents (e-prop)

- `numerical_verification_eprop_factorization_vs_BPTT.py`: This script verifies numerically the equation (1) of the paper.  

- `e-prop_tutorials_Figure3_and_S7`: Code for the temporal credit assignment (Fig. 3) task and the pattern generation task (Fig. 7).
Also, following section S1.1 from the paper, e-prop can be implemented easily with automatic differentiation.
In the folder we demonstrate the mathematical equivalence between a straight forward implementation of e-prop and an alternative that uses more efficiently the features of tensorflow (see `stop_gradient` parameter in the cell objects).
For TIMIT and ATARI with publish the more efficient implementations.

- `e_prop_tutorials_Figure2_and_S2_S3_S4`: Code to reproduce the results on the phoneme recognition task.

- `Figure_4_and_5_ATARI`: Code for reward based e-prop on ATARI games.  
