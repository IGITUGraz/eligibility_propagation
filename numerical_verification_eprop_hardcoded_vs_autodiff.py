# Copyright 2020, the e-prop team
# Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons
# Authors: G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass
#
# In this script we verify that the computation of symmetric e-prop derived in the paper
# equal to what we compute the auto-diff version of e-prop.
#
# This numerical verification relies on a single script structured as follows:
# 1. Let's define some parameters
# 2. Define the network model and the inputs
# 3. We simulate the network.
# 4. Define the learning signal with equation (4) for an arbitrary loss function
# 5. Compute the gradients following the online definition of eligibility traces for ALIF equation (25)
# 6. Compute the gradients with auto-diff (with the cell parameter "stop_gradient=True" it leads to e-prop)
# 7. Start the tensorflow session to run the computation
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
T = 10  # ms
f0 = 100  # Hz

thr = 0.62
beta = 0.07 * np.concatenate([np.zeros(n_LIF), np.ones(n_ALIF)])
dampening_factor = 0.3
n_ref = 3

# 2. Define the network model and the inputs
cell = EligALIF(n_in=n_in, n_rec=n_LIF + n_ALIF, tau=tau_v, beta=beta, thr=thr,
                dt=dt, tau_adaptation=tau_a, dampening_factor=dampening_factor,
                stop_z_gradients=True,  # This option makes it possible to blindly compute e-prop with auto-diff.
                n_refractory=n_ref)

# Define an arbitrary input, here some poisson spike train.
inputs = tf.random_uniform(shape=[1, T, n_in]) < f0 * dt / 1000
inputs = tf.cast(inputs, tf.float32)

# 3. We simulate the network.
spikes = []
voltages = []
thr_variations = []

state = cell.zero_state(1, tf.float32, n_rec=n_rec)
for t in range(T):
    outs, state = cell(inputs[:, t], state)
    spikes_t, hidden_states_t = outs

    spikes.append(spikes_t)
    voltages.append(hidden_states_t[..., 0])
    thr_variations.append(hidden_states_t[..., 1])

# Stack the lists as tensors (second dimension is time)
# - spikes and learning signals will have shape: [n_batch, n_time , n_neuron]
# - eligibility traces will have shape: [n_batch, n_time , n_neuron, n_neuron]
spikes = tf.stack(spikes, axis=1)
voltages = tf.stack(voltages, axis=1)
thr_variations = tf.stack(thr_variations, axis=1)

# 4. Define the learning signal with eqution (4) for an arbitrary loss function
# (here regression with a random target)
w_out = tf.random_normal(shape=[n_rec, 1])
decay_out = tf.exp(-1 / 20)
z_filtered = exp_convolve(spikes, decay_out)
y_out = tf.einsum("btj,jk->btk", z_filtered, w_out)
y_target = tf.random_normal(shape=[1, T, 1])
loss = 0.5 * tf.reduce_sum((y_out - y_target) ** 2)

# This defines the true learning signal as in equation (4)
# Einsum performs a tensor multiplication with more flexibility on the combination of indices.
learning_signals = tf.einsum("btk,jk->btj", y_out - y_target, w_out)

# 5. Compute the gradients with cell.compute_loss_gradient(...),
# following the online definition of eligibility traces for ALIF equation (25)
# the gradients with e-prop are computed with equation (1) of the paper
pre_synpatic_spike_one_step_before = shift_by_one_time_step(spikes)
gradients_eprop, eligibility_traces, _, _ = \
    cell.compute_loss_gradient(learning_signals, pre_synpatic_spike_one_step_before, spikes, voltages,
                               thr_variations, decay_out, True)

# 6. Compute the gradients with auto-diff as a ground truth (stop_z_gradients=True)
gradients_autodiff = tf.gradients(loss, cell.w_rec_var)[0]

# 7. Start the tensorflow session to run the computation
# (until now we only built a computational graph, no simulation has been performed)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf_tensors = {'inputs': inputs,
              'spikes': spikes,
              'gradients_eprop': gradients_eprop,
              'gradients_autodiff': gradients_autodiff,
              'eligibility_traces': eligibility_traces,
              'y_out': y_out,
              'y_target': y_target,
              'learning_signals': learning_signals}

np_tensors = sess.run(tf_tensors)

# Show the gradients
fig, ax_list = plt.subplots(1, 2)
ax_list[0].imshow(np_tensors['gradients_eprop'])
ax_list[0].set_title("Gradient dE/dW_ji with e-prop")
ax_list[1].imshow(np_tensors['gradients_autodiff'])
ax_list[1].set_title("Gradient dE/dW_ji with autodiff")

# Compute the relative error:
g_e_prop = np_tensors['gradients_eprop']
g_autodiff = np_tensors['gradients_autodiff']
M = np.max(np.abs(g_autodiff))

print("Max abs value of the true gradient: ", M)
assert (not np.any(np.isnan(g_autodiff)), "The auto-diff has NaN coeffs, this not a very interesting verification.")
assert M != 0, "The auto-diff gradient is zero, this not a very interesting verification."
g_e_prop /= M
g_autodiff /= M

gradient_errors = (g_e_prop - g_autodiff) ** 2
max_gradient_errors = np.max(gradient_errors)
print("Gradients computed with symmetric e-prop:")
print(np.array_str(np_tensors['gradients_eprop'], precision=5, suppress_small=True))
print("Gradients computed with autodiff (and \"stop_gradient=True\"):")
print(np.array_str(np_tensors['gradients_autodiff'], precision=5, suppress_small=True))
print("Maximum element wise errors: {}".format(max_gradient_errors))

# Some plots to visualize what is happening.
fig, ax_list = plt.subplots(4, figsize=(8, 12), sharex=True)
raster_plot(ax_list[0], np_tensors['inputs'][0])
ax_list[0].set_ylabel("Input spikes")

raster_plot(ax_list[1], np_tensors['spikes'][0])
ax_list[1].set_ylabel("Spikes")

v_max = np.max(np.abs(np_tensors['learning_signals']))
ax_list[2].pcolor(np.arange(T), np.arange(n_rec), np_tensors['learning_signals'][0].T, cmap='seismic', vmin=-1, vmax=1)
ax_list[2].set_ylabel("Learning signals")

for i in range(3):
    for j in range(3):
        if i != j:
            ax_list[3].plot(np.arange(T), np_tensors['eligibility_traces'][0, :, i, j])
ax_list[3].set_ylabel("Eligibility traces")
ax_list[3].set_xlabel("time in ms")

plt.show()
