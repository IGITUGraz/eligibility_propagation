import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import numpy.random as rd
import json

class NumpyAwareEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyAwareEncoder, self).default(obj)



def raster_plot(ax,spikes,linewidth=0.8,**kwargs):

    n_t,n_n = spikes.shape
    event_times,event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]

    for n,t in zip(event_ids,event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)

    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([0, n_t])
    ax.set_yticks([0, n_n])

def strip_right_top_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()



def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None): rng = rd.RandomState(freezing_seed)
    else: rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes



def generate_click_task_data(batch_size, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
                             n_cues=7, t_cue=100, t_interval=150,
                             n_input_symbols=4):
    t_seq = seq_len
    n_channel = n_neuron // n_input_symbols

    # randomly assign group A and B
    prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
    idx = rd.choice([0, 1], batch_size)
    probs = np.zeros((batch_size, 2), dtype=np.float32)
    # assign input spike probabilities
    probs[:, 0] = prob_choices[idx]
    probs[:, 1] = prob_choices[1 - idx]

    cue_assignments = np.zeros((batch_size, n_cues), dtype=np.int)
    # for each example in batch, draw which cues are going to be active (left or right)
    for b in range(batch_size):
        cue_assignments[b, :] = rd.choice([0, 1], n_cues, p=probs[b])

    # generate input nums - 0: left, 1: right, 2:recall, 3:background noise
    input_nums = 3*np.ones((batch_size, seq_len), dtype=np.int)
    input_nums[:, :n_cues] = cue_assignments
    input_nums[:, -1] = 2

    # generate input spikes
    input_spike_prob = np.zeros((batch_size, t_seq, n_neuron))
    d_silence = t_interval - t_cue
    for b in range(batch_size):
        for k in range(n_cues):
            # input channels only fire when they are selected (left or right)
            c = cue_assignments[b, k]
            # reverse order of cues
            #idx = sequence_length - int(recall_cue) - k - 1
            idx = k
            input_spike_prob[b, d_silence+idx*t_interval:d_silence+idx*t_interval+t_cue, c*n_channel:(c+1)*n_channel] = f0

    # recall cue
    input_spike_prob[:, -recall_duration:, 2*n_channel:3*n_channel] = f0
    # background noise
    input_spike_prob[:, :, 3*n_channel:] = f0/4.
    input_spikes = generate_poisson_noise_np(input_spike_prob)

    # generate targets
    target_mask = np.zeros((batch_size, seq_len), dtype=np.bool)
    target_mask[:, -1] = True
    target_nums = np.zeros((batch_size, seq_len), dtype=np.int)
    target_nums[:, :] = np.transpose(np.tile(np.sum(cue_assignments, axis=1) > int(n_cues/2), (seq_len, 1)))

    return input_spikes, input_nums, target_nums, target_mask


# raster plot
def update_plot(plot_result_values, ax_list, plot_traces=False, batch=0, n_max_neuron_per_raster=10, title=None,
                eps_sel=None, trace_sel=None):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    flags = plot_result_values['flags']
    n_rec = flags['n_regular']
    n_con = flags['n_adaptive']
    n_input_symbols = 4
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)
        ax_list[0].set_title(title)
    k_ax = 0
    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    for data, d_name in zip([plot_result_values['input_spikes'], plot_result_values['z'], plot_result_values['z']],
                            ['Input', 'LIF', 'ALIF']):
        if np.size(data) > 0:
            ax = ax_list[k_ax]
            data = data[batch]

            if d_name == 'Input':
                n_max = data.shape[1]
                cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
                # Value 2 row

            elif d_name == 'LIF':
                n_max = min(n_rec, n_max_neuron_per_raster)
                cell_select = np.linspace(start=0, stop=n_rec - 1, num=n_max, dtype=int)

            elif d_name == 'ALIF':
                n_max = min(n_con, n_max_neuron_per_raster)
                cell_select = np.linspace(start=n_rec, stop=n_rec + n_con - 1, num=n_max, dtype=int)

            if cell_select.size is not 0:
                k_ax += 1

                data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
                if d_name == 'Input':
                    n_channel = data.shape[1] // n_input_symbols
                    # insert empty row
                    zero_fill = np.zeros((data.shape[0], int(n_channel / 2)))
                    data = np.concatenate((data[:, 3 * n_channel:], zero_fill,
                                           data[:, 2 * n_channel:3 * n_channel], zero_fill,
                                           data[:, :n_channel], zero_fill,
                                           data[:, n_channel:2 * n_channel]), axis=1)
                    ax.set_yticklabels([])
                    ax.add_patch(  # Value 0 row
                        patches.Rectangle((0, 2 * n_channel + 2 * int(n_channel / 2)), data.shape[0], n_channel,
                                          facecolor="red", alpha=0.1))
                    ax.add_patch(  # Value 1 row
                        patches.Rectangle((0, 3 * n_channel + 3 * int(n_channel / 2)), data.shape[0], n_channel,
                                          facecolor="blue", alpha=0.1))

                raster_plot(ax, data, linewidth=0.4)
                ax.set_ylabel(d_name)
                ax.set_xticklabels([])
                ax.set_xticks([])

    ax = ax_list[k_ax]
    output2 = plot_result_values['out_plot'][batch, :, 1]

    presentation_steps = np.arange(output2.shape[0])

    # ax.add_patch(patches.Rectangle((output2.shape[0] - FLAGS.t_cue, 0),
    #                               FLAGS.t_cue, 1, facecolor="green", alpha=0.15))

    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel('Output')
    line_output2, = ax.plot(presentation_steps, output2, color='purple', label='softmax', alpha=0.7)
    line_base, = ax.plot(presentation_steps, np.ones_like(presentation_steps) * 0.5, '--', color='grey',
                         label='decision threshold', alpha=0.7)

    ax.axis([0, presentation_steps[-1] + 1, -0.3, 1.1])
    ax.legend(handles=[line_output2, line_base], loc='lower center', fontsize=7,
              bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.set_xticklabels([])
    ax.set_xticks([])

    # plot learning signal
    if plot_traces:
        k_ax += 1
        ax = ax_list[k_ax]
        ax.set_ylabel('$L_j$')
        sub_data = plot_result_values['z_grad'][batch]

        presentation_steps = np.arange(sub_data.shape[0])
        cell_select = np.linspace(start=0, stop=99, num=10, dtype=int)
        # ax.plot(sub_data[:, cell_select], label='learning_signal', alpha=1, linewidth=1)
        v = np.maximum(abs(np.min(sub_data[:, cell_select].T)), abs(np.max(sub_data[:, cell_select].T)))
        p = ax.pcolor(sub_data[:, cell_select].T, label='learning_signal', cmap='seismic', alpha=0.4, linewidth=0.3,
                      vmin=-v, vmax=v)
        ax.set_xticklabels([])
        ax.set_xticks([])

        # plot eligibility traces
        k_ax += 1
        ax = ax_list[k_ax]
        # ax.spines['bottom'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_xticks([])

        e_trace = plot_result_values['e_trace'][batch]
        epsilon = plot_result_values['epsilon_a'][batch]

        presentation_steps = np.arange(e_trace.shape[0])
        if trace_sel is None:
            trace_sel = np.linspace(start=0, stop=e_trace.shape[1], num=n_max)
        if eps_sel is None:
            eps_sel = np.linspace(start=0, stop=epsilon.shape[1], num=n_max)

        colors = plt.get_cmap("tab10")

        for k in range(e_trace.shape[1]):
            if k in trace_sel:
                # ax.plot(e_trace_filtered[:, k], alpha=0.8, linewidth=1, label=str(k), color=colors(k))
                ax.plot(e_trace[:, k], alpha=0.8, linewidth=1, label=str(k), color=colors(k))

        ax.axis([0, presentation_steps[-1], 1.2 * np.min(e_trace), np.max(e_trace)])  # [xmin, xmax, ymin, ymax]
        ax.set_ylabel('e-trace')
        # ax.legend()

        # plot epsilon
        k_ax += 1
        ax = ax_list[k_ax]

        for k in range(epsilon.shape[1]):
            if k in eps_sel:
                ax.plot(epsilon[:, k], label='slow e-trace component', alpha=0.8, linewidth=1, color=colors(k))

        ax.axis([0, presentation_steps[-1], np.min(epsilon), np.max(epsilon)])  # [xmin, xmax, ymin, ymax]
        #     ax.set_xticks(np.linspace(0, epsilon.shape[0], 5))
        #ax.set_xticks([0, 500, 1000, 1500, 2000])

        ax.set_ylabel('slow factor')

    ax.set_xlabel('Time in ms')

    plt.subplots_adjust(hspace=0.3)
