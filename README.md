## Tutorial on eligibility propagation
Full paper: https://arxiv.org/abs/1901.09049  
Authors: Guillaume Bellec\*, Franz Scherr\*, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass

The present tutorial is built as a single short script which trains a recurrent network (RNN) of Leaky Integrate and Fire (LIF) neurons
using the eligibility propagation (e-prop) algorithm.

This code is written and tested with Tensorflow 1.12.0, numpy 1.15.4 and python 3.6.7.
When running `tutorial.py` with default parameters one should obtain after 1000 iterations (2 minutes on a laptop) the following raster plot.

![Raster](./figures/raster.jpg)

From top to bottom, each row represents respectively:
the input spikes, the network spikes, the output (in orange) and the target (dashed blue) signals,
the pre-synaptic term of e-prop, the post synaptic term of e-prop, the final eligibility traces and the learning signals.
