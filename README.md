Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons  
Authors: G Bellec\*, F Scherr\*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass  
[[Bioarxiv]](https://www.biorxiv.org/content/10.1101/738385v3)  
[[Older pre-print with related content]](https://arxiv.org/abs/1901.09049)  

## Table of contents (e-prop)

- The script `numerical_verification_eprop_factorization_vs_BPTT.py` verifies numerically the equation (1) of the paper.  

- Following section S1.1 from the paper, e-prop can be implemented easily with automatic differentiation.
In the folder `e-prop_tutorials_Figure3_and_S7` we demonstrate the mathematical equivalence between an online implementation of e-prop and an alternative that is more efficiently implemented in tensorflow.

- The folder `Figure_2_TIMIT` provides the code to reproduce all the simulation results on the phoneme recognition task.  

- The folder `Figure_4_and_5_ATARI` provides the code to reproduce all our reward with reward based e-prop on ATARI games.  