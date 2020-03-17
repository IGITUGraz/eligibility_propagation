# In this check we verify numerically the exactness of the equation (1) of the paper:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from e_prop_tutorials_Figure3_and_S7.models import EligALIF
from e_prop_tutorials_Figure3_and_S7.tools import raster_plot

# 1. Let's define some parameters
n_in = 3
n_LIF = 5
n_ALIF = 5
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
                stop_z_gradients=False, n_refractory=n_ref)

inputs = tf.random_uniform(shape=[1,T,n_in]) < f0 * dt / 1000
inputs = tf.cast(inputs, tf.float32)

# 3. This way of simulating the network is not efficient in tensorflow
# We do this here to be able to compute the true learning signals: dE/dz
spikes = []
hidden_states = []
state = cell.zero_state(1,tf.float32,n_rec=n_rec)
for t in range(T):
    outs, state = cell(inputs[:,t],state)
    spikes.append(outs[0])
    hidden_states.append(outs[1])

mean_firing_rates = tf.reduce_sum(spikes,axis=(0,1)) / (T /1000)
loss = tf.reduce_sum((mean_firing_rates - f0)**2)

# This defines the true learning signal: dE/dz
dE_dz = []
for t in range(T):
    L = tf.gradients(loss,spikes[t])[0]
    dE_dz.append(L)

# concatenate the lists as tensors
# spikes and learning signals will be: [n_batch, n_time , n_neuron]
# eligibility traces are: [n_batch, n_time , n_neuron, n_neuron]
spikes = tf.stack(spikes,axis=1)
hidden_states = tf.stack(hidden_states,axis=1)
dE_dz = tf.stack(dE_dz,axis=1)

# 5. compute the eligibility traces:
v_scaled = cell.compute_v_relative_to_threshold_values(hidden_states)

spikes_last_time_step = tf.concat([tf.zeros_like(spikes[:,0:1]),spikes[:,:-1]],axis=1)
eligibility_traces, _, _, _ = cell.compute_eligibility_traces(v_scaled, spikes_last_time_step, spikes, True)

# 6. Compute the gradients, we will compare e-prop and BPTT:
# the gradients with e-prop are computed with equation (1) from the paper
gradients_eprop = tf.einsum('btj,btij->ij',dE_dz,eligibility_traces)
gradients_BPTT = tf.gradients(loss,cell.w_rec_var)[0]

# 7. Initialize the tensorflow session to compute the tensor values.
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
print("Gradients computed with e-prop:")
print(np.array_str(np_tensors['gradients_eprop'],precision=3,suppress_small=True))
print("Gradients computed with BPTT:")
print(np.array_str(np_tensors['gradients_BPTT'],precision=3,suppress_small=True))
print("Maximum element wise errors: {}".format(max_gradient_errors))

# Open an interactive matplotlib window to plot in real time
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



