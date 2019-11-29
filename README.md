Full paper: https://arxiv.org/abs/1901.09049  
Authors: Guillaume Bellec\*, Franz Scherr\*, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass

## Tutorials on eligibility propagation (e-prop)

The present repository gathers two tutorials. In the first tutorial  `tutorial_pattern_generation.py`, one trains a recurrent network (RNN) of Leaky Integrate and Fire (LIF) neurons using the eligibility propagation (e-prop) algorithm on a pattern generation task. This task does not require working memory.

In the second tutorial `tutorial_evidence_accumulation_with_alif.py` an evidence accumulation task is considered and it requires to remember over a long delay, hence we equiped the neuron model with an adaptive threshold with a longer time constant. The eligibility traces that result in e-prop with this adaptive neuron model are richer.

This code is written and tested with tensorflow 1.12.0, numpy 1.17.3 and python 3.6.9. The below figure was obtained by running:  
```tutorial_evidence_accumulation_with_alif.py -feedback random -eprop -eprop_impl hardcoded -n_batch 32```  
When running the hardcoded implementation of e-prop on a GPU and encountering an OOM error, it is recommended to reduce the batch size to 16 or 8. The early stopping criterion should be reached after around 200 - 500 iterations. 

<img src="./figures/evidence_acc_training.png"
     alt="Raster plot"
     style="width: 200;" />
