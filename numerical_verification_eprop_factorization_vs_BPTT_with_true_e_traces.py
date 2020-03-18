# In this check we verify numerically the exactness of the equation (1) of the paper:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from e_prop_tutorials_Figure3_and_S7.models import EligALIF, exp_convolve, shift_by_one_time_step
from e_prop_tutorials_Figure3_and_S7.tools import raster_plot

# 1. Let's define some parameters
n_in = 3
n_LIF = 0
n_ALIF = 5
n_rec = n_ALIF + n_LIF

dt = 1  # ms
tau_v = 20  # ms
tau_a = 500  # ms
T = 15  # ms

f0 = 100  # Hz

thr = 0.62
beta = 0.07 * np.concatenate([np.zeros(n_LIF),np.ones(n_ALIF)])
dampening_factor = 0.3
n_ref = 2

# 2. Define the network model and the inputs
cell = EligALIF(n_in=n_in, n_rec=n_LIF + n_ALIF, tau=tau_v, beta=beta, thr=thr,
                dt=dt, tau_adaptation=tau_a, dampening_factor=dampening_factor,
                stop_z_gradients=False, n_refractory=n_ref)

inputs = np.zeros([1,T,n_in])
inputs[0,1,0] = 1.
#inputs = tf.random_uniform(shape=[1,T,n_in]) < f0 * dt / 1000
inputs = tf.cast(inputs, tf.float32)

w_in_0 = np.zeros((n_in,n_rec))
w_in_0[0,0] = 1.

w_rec_0 = np.zeros((n_rec,n_rec))
w_rec_0[0,1] = 1.
w_rec_0[1,2] = 1.
w_rec_0[2,3] = 1.
w_rec_0[3,4] = 1.
w_rec_0[2,0] = 1.

set_weights = tf.group(tf.assign(cell.w_in_var,w_in_0),tf.assign(cell.w_rec_var,w_rec_0))


# 3. This way of simulating the network is not efficient in tensorflow
# We a for loop instead of using tf.nn.dynamic_rnn(...) to compute the true learning signals: dE/dz (total derivative)
spikes = []
spikes_stopped = []
hidden_states = []
hidden_states_stopped = []
state = cell.zero_state(1,tf.float32,n_rec=n_rec)
state_stopped = cell.zero_state(1,tf.float32,n_rec=n_rec)
for t in range(T):
    outs_stopped, state_stopped = cell(inputs[:,t],state_stopped, stop_gradient=True)
    outs, state = cell(inputs[:,t],state)
    spikes.append(outs[0])
    spikes_stopped.append(outs_stopped[0])
    hidden_states.append(outs[1])
    hidden_states_stopped.append(outs_stopped[1])

w_out = tf.ones(shape=[n_rec,1])
#decay_out = tf.exp(-1/20)
#z_filtered = exp_convolve(tf.stack(spikes,1),decay_out)
z_filtered = spikes
y_out = tf.matmul(z_filtered,w_out)
y_target = tf.constant(np.abs(np.arange(T) - T/2).reshape((1,T,1)),dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y_out - y_target))

# mean_firing_rates = tf.reduce_sum(spikes,axis=(0,1)) / (T /1000)
# loss = tf.reduce_sum((mean_firing_rates - f0)**2)

# This defines the true learning signal: dE/dz (total derivative)
dE_dz = tf.gradients(loss, [spikes[t] for t in range(T)])

# Stack the lists as tensors (second dimension is time)
# - spikes and learning signals will have shape: [n_batch, n_time , n_neuron]
# - eligibility traces will have shape: [n_batch, n_time , n_neuron, n_neuron]
spikes = tf.stack(spikes,axis=1)
hidden_states = tf.stack(hidden_states,axis=1)
dE_dz = tf.stack(dE_dz,axis=1)

hidden_states_stopped = tf.stack(hidden_states_stopped,axis=1)

def get_gradient_wrt_w(z):
    g = tf.gradients(z,cell.w_rec_var)[0]
    if g is None:
        return tf.zeros_like(cell.w_rec_var)
    else:
        return g

true_eligibility_traces = [get_gradient_wrt_w(z) for z in spikes_stopped]
true_eligibility_traces = tf.stack(true_eligibility_traces,axis=0)[None, ...]

# 5. compute the eligibility traces:
v_scaled = cell.compute_v_relative_to_threshold_values(hidden_states)
spikes_last_time_step = shift_by_one_time_step(spikes)
eligibility_traces, _, _, _ = cell.compute_eligibility_traces(v_scaled, spikes_last_time_step, spikes, True)


# 6. Compute the gradients, we will compare e-prop and BPTT:
# the gradients with e-prop are computed with equation (1) from the paper
gradients_eprop = tf.einsum('btj,btij->ij',dE_dz,true_eligibility_traces)
#gradients_eprop = tf.einsum('btj,btij->ij',dE_dz,eligibility_traces)
gradients_BPTT = tf.gradients(loss,cell.w_rec_var)[0]

# 7. Initialize the tensorflow session to compute the tensor values.
# (until now we only built a computational graph, no simulation has been performed)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

sess.run(set_weights)

tf_tensors = {'inputs': inputs,
              'spikes': spikes,
              'voltages': hidden_states[..., 0],
              'voltages_stopped': hidden_states_stopped[..., 0],
              'gradients_eprop': gradients_eprop,
              'gradients_BPTT': gradients_BPTT,
              'eligibility_traces': eligibility_traces,
              'true_eligibility_traces': true_eligibility_traces,
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

e_trace_hardcoded = np_tensors['eligibility_traces']
e_trace_autodiff = np_tensors['true_eligibility_traces']
v_stopped = np_tensors['voltages_stopped']
v = np_tensors['voltages']
for t in range(T):
    elig_error = (e_trace_autodiff[0,t,...] - e_trace_hardcoded[0,t,...])**2
    NN = np.max(e_trace_autodiff[0,t,...]**2)
    max_elig_error = np.max(elig_error / NN)
    print("t:",t,' e trace error: {}'.format(max_elig_error))

    v_error = (v_stopped[0,t,...] - v[0,t,...])**2
    NN = np.max(v[0,t,...]**2)
    max_v_error = np.max(v_error / NN)
    print(' v error: {}'.format(max_v_error))

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



