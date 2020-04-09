# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass
#
# In this check we verify numerically the exactness of the equation (1) of the paper.
# In other word, that the loss gradient dE/dW_ji can be computed with the formula:
#
# dE/dW_ji = sum_t dE/dz_j(t) [d z/dW_ji (t) ]_local
#
# As a ground it will be computed to the gradient computed with BPTT via auto-differentiation.
# This numerical verification is contained in a single script and follows the structure:
#
# 1. Let's define some parameters
# 2. Define the network model and the inputs
# 3. Simulate the network
# 4. Compute the true learning signal: dE/dz (total derivative) for an arbitrary loss function
# 5. Compute the eligibility traces for ALIFs using formula (25)
# 6. Compute the gradients with equation (1)
# 7. Compute the gradients given by BPTT using auto-diff
# 8. Start the tensorflow session and compute numerical verification.
#
# The relative difference between the two resulting gradients dE/dW_ij are approximately 10^-14.
# This tiny difference is the expected machine precision for two different computation schemes of the same gradient.
#
# This script requires was tested with tensorflow 1.15 and python3.6.
# More details requirements are explained in the folder Figure_3_and_S7_...


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Figure_3_and_S7_e_prop_tutorials.models import EligALIF, exp_convolve, shift_by_one_time_step
from Figure_3_and_S7_e_prop_tutorials.tools import raster_plot

# 1. Let's define some parameters
n_in = 3
n_LIF = 4
n_ALIF = 4
n_rec = n_ALIF + n_LIF

dt = 1  # ms
tau_v = 20  # ms
tau_a = 500  # ms
T = 50  # ms
f0 = 100  # Hz

thr = 0.62
beta = 0.07 * np.concatenate([np.zeros(n_LIF),np.ones(n_ALIF)])
dampening_factor = 0.3
n_ref = 2

# 2. Define the network model and the inputs
cell = EligALIF(n_in=n_in, n_rec=n_LIF + n_ALIF, tau=tau_v, beta=beta, thr=thr,
                dt=dt, tau_adaptation=tau_a, dampening_factor=dampening_factor,
                stop_z_gradients=False,  # here it computes the BPTT gradients, set it to True to compute instead e-prop with auto-diff
                n_refractory=n_ref)

inputs = tf.random_uniform(shape=[1,T,n_in]) < f0 * dt / 1000
inputs = tf.cast(inputs, tf.float32)

# 3. Simulate the network,
# Using a for loop instead of using tf.nn.dynamic_rnn(...) is not efficient
# belows allows us to compute the true learning signals: dE/dz (total derivative)
# with auto-diff to perform the numerical verification
spikes = []
hidden_states = []
state = cell.zero_state(1,tf.float32,n_rec=n_rec)
for t in range(T):
    outs, state = cell(inputs[:,t],state)
    spikes.append(outs[0])
    hidden_states.append(outs[1])

# 4. Compute the true learning signal: dE/dz (total derivative) for an arbitrary loss function
# (here a regression with a random signal)
w_out = tf.random_normal(shape=[n_rec,1])
decay_out = tf.exp(-1/20)
z_filtered = exp_convolve(tf.stack(spikes,1),decay_out)
y_out = tf.matmul(z_filtered,w_out)
y_target = tf.random_normal(shape=[1,T,1])
loss = tf.reduce_mean(tf.square(y_out - y_target))

# This defines the true learning signal: dE/dz (total derivative)
dE_dz = tf.gradients(loss, [spikes[t] for t in range(T)])

# Stack the lists as tensors (second dimension is time)
# - spikes and learning signals will have shape: [n_batch, n_time , n_neuron]
# - eligibility traces will have shape: [n_batch, n_time , n_neuron, n_neuron]
spikes = tf.stack(spikes,axis=1)
hidden_states = tf.stack(hidden_states,axis=1)
dE_dz = tf.stack(dE_dz,axis=1)

# 5. Compute the eligibility traces for ALIFs using formula (25)
v_scaled = cell.compute_v_relative_to_threshold_values(hidden_states)
spikes_last_time_step = shift_by_one_time_step(spikes)
eligibility_traces, _, _, _ = cell.compute_eligibility_traces(v_scaled, spikes_last_time_step, spikes, True)

# 6. Compute the gradients with equation (1)
gradients_eprop = tf.einsum('btj,btij->ij',dE_dz,eligibility_traces)

# 7. Compute the gradients given by BPTT using auto-diff
gradients_BPTT = tf.gradients(loss,cell.w_rec_var)[0]

# 8. Start the tensorflow session and compute numerical verification.
# (until now we only built a computational graph, no simulation has been performed)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf_tensors = {'inputs': inputs,
              'spikes': spikes,
              'gradients_eprop': gradients_eprop,
              'gradients_BPTT': gradients_BPTT,
              'eligibility_traces': eligibility_traces,
              'learning_signals': dE_dz}

np_tensors = sess.run(tf_tensors)

# Show the gradients
fig, ax_list = plt.subplots(1,2)
ax_list[0].imshow(np_tensors['gradients_eprop'])
ax_list[0].set_title("Gradient dE/dW_ji with e-prop")
ax_list[1].imshow(np_tensors['gradients_BPTT'])
ax_list[1].set_title("Gradient dE/dW_ji with BPTT")

# Compute the relative error:
g_e_prop = np_tensors['gradients_eprop']
g_bptt = np_tensors['gradients_BPTT']
M = np.max(np.abs(g_bptt))
g_e_prop /= M
g_bptt /= M

gradient_errors = (g_e_prop - g_bptt)**2
max_gradient_errors = np.max(gradient_errors)
print("Gradients computed with the e-prop factorization (equation (1) in the paper):")
print(np.array_str(np_tensors['gradients_eprop'],precision=3,suppress_small=True))
print("Gradients computed with BPTT and auto-differentiation:")
print(np.array_str(np_tensors['gradients_BPTT'],precision=3,suppress_small=True))
print("Maximum element wise errors: {}".format(max_gradient_errors))

fig, ax_list = plt.subplots(4, figsize=(8, 12), sharex=True)
raster_plot(ax_list[0],np_tensors['inputs'][0])
ax_list[0].set_ylabel("Input spikes")

raster_plot(ax_list[1],np_tensors['spikes'][0])
ax_list[1].set_ylabel("Spikes")

v_max = np.max(np.abs(np_tensors['learning_signals']))
ax_list[2].pcolor(np.arange(T),np.arange(n_rec),np_tensors['learning_signals'][0].T,cmap='seismic',vmin=-1,vmax=1)
ax_list[2].set_ylabel("Learning signals")

for i in range(3):
    for j in range(3):
        if i != j:
            ax_list[3].plot(np.arange(T),np_tensors['eligibility_traces'][0,:,i,j])
ax_list[3].set_ylabel("Eligibility traces")

plt.show()



