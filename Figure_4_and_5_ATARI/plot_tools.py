import matplotlib.pyplot as plt
import numpy as np


def _prep_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def update_traj_plot(axes, data, bootstrap, zs, hs, gamma, n_supersampling=1, beta=.1, thr=10., level_name='atari'):
    for ax in axes:
        ax.clear()
        _prep_ax(ax)
    rewards = []
    dones = []
    baselines = []
    observations = []
    policy_logits = []
    action_probabilies = []
    for d in data:
        rewards.append(d.env_outputs.reward)
        dones.append(d.env_outputs.done)
        baselines.append(d.agent_outputs.baseline)
        observations.append(d.env_outputs.observation[0])
        policy_logits.append(d.agent_outputs.policy_logits)
        action_probabilies.append(d.action_probabilities)
    rewards = np.concatenate(rewards, 0)
    dones = np.concatenate(dones, 0)
    try:
        zs = np.concatenate(zs, 0)
        hs = np.concatenate(hs, 0)
    except ValueError as e:
        print(e)
        pass
    value_target = np.zeros_like(rewards)
    value_target[-1] = (1 - dones[-1]) * gamma * bootstrap + rewards[-1]
    N = rewards.shape[0]
    for i in range(N - 1):
        t = (1 - dones[N - i - 2]) * gamma * value_target[N - i - 1] + rewards[N - i - 2]
        value_target[N - i - 2] = t
    baselines = np.concatenate(baselines, 0)
    action_probabilies = np.concatenate(action_probabilies, 0)

    x = np.arange(action_probabilies.shape[0])

    axes[0].plot(x, value_target, 'k', lw=2, alpha=.7, label='Cumulative reward')
    axes[0].plot(x, baselines[:], 'b', lw=2, alpha=.7, label='Prediction')
    axes[0].set_ylabel('Value\ntarget', fontsize=8)
    axes[0].legend(ncol=1, fontsize=7, loc=(1.01, .3), frameon=False)

    cum_probabilies = np.cumsum(action_probabilies, -1)
    x = np.arange(cum_probabilies.shape[0])
    if level_name.count('pong') > 0:
        labels = ['Stay',
                  'Up',
                  'Down']
    else:
        labels = [
            'No op', 'Fire', 'Up', 'Right', 'Left', 'Down',
            'Up-right', 'Up-Left', 'Down-right', 'Down-left', 'Up-fire', 'Right-fire',
            'Left-fire', 'Down-fire', 'Up-right-fire', 'Up-left-fire', 'Down-right-fire', 'Down-left-fire'
        ]
    cum_probabilies = np.concatenate((np.zeros_like(cum_probabilies[:, :1]), cum_probabilies), -1)
    colors = ['lightgray', 'mediumvioletred', 'lightblue', 'C1', 'mediumaquamarine', 'royalblue', 'navy', 'darkviolet', 'k'] * 2
    for i in range(cum_probabilies.shape[1] - 1, 0, -1):
        axes[1].plot(x, action_probabilies[:, i - 1], lw=1, alpha=.7, color=colors[(i - 1) % len(colors)], label=labels[i - 1])
    axes[1].legend(ncol=1, fontsize=7, loc=(1.01, .05), frameon=False)
    axes[1].set_yticks([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Action\nprobabilities', fontsize=8)
    axes[5].set_xlabel('Time in ms', fontsize=8)
    axes[0].set_xlim([0, x.shape[0]])
    axes[1].set_xlim([0, x.shape[0]])

    n_neurons_max = 10

    inds = beta >= 1e-5
    p = axes[2].pcolormesh(zs[:, np.logical_not(inds)][:, :n_neurons_max].T, cmap='Greys')
    axes[2].set_ylabel('LIF', fontsize=8)
    axes[2].set_yticks([0, zs[:, inds][:, :n_neurons_max].shape[1]])
    p = axes[3].pcolormesh(zs[:, inds][:, :n_neurons_max].T, cmap='Greys')
    axes[3].set_ylabel('ALIF', fontsize=8)
    axes[3].set_yticks([0, zs[:, inds][:, :n_neurons_max].shape[1]])
    athr = hs[:, inds, 0][:, :n_neurons_max] - hs[:, inds, 1][:, :n_neurons_max] * beta[inds][None, :n_neurons_max]
    p = axes[4].plot(athr, color='b', alpha=.5, lw=.5)
    axes[4].set_ylabel('Voltages (ALIF)', fontsize=8)
    axes[4].set_xlim([0, athr.shape[0]])

    athr = (hs[:, inds, 1] + thr)[:, :n_neurons_max]
    abs_max = np.max(athr)
    p = axes[5].plot(athr, color='b', alpha=.5, lw=.5)
    axes[5].set_ylim([0, abs_max])
    axes[5].set_ylabel('Thresholds', fontsize=8)
    axes[5].set_xlim([0, athr.shape[0]])

    for ax in axes[:-1]:
        ax.set_xticks([])

    done_inds, = np.where(dones)
    for ax_id, ax in enumerate(axes):
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ylim = ax.get_ylim()
        for di in done_inds:
            if ax_id in [2, 3, 4, 5]:
                di = (di + 1) * n_supersampling - 1
            ax.plot([di + 1, di + 1], ylim, 'k--', alpha=.5)
        ax.set_ylim(ylim)
        ax.yaxis.set_label_coords(-.065, .5)

    plt.subplots_adjust(right=.8, hspace=.3)


def update_performance_plot(ax, data):
    x = np.array(data['x'])
    y = np.array(data['y'])
    ystd = np.array(data['ystd'])

    ax.clear()
    ax.fill_between(x, y - ystd, y + ystd, facecolor='b', alpha=.1)
    ax.plot(x, y, 'b', lw=2, alpha=.8)

    ax.set_ylabel('Episode return', fontsize=12)
    ax.set_xlabel('Environment frames', fontsize=12)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
