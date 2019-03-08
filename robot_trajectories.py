import numpy as np


class RobotTrajectories(object):
    def __init__(self, n_batch, seq_length, n_periods, dt_step=.001, sine_seed=3000, static=False):
        # self.period_range = ((250., 250.), (500., 500.), (250., 250.))
        # self.weight_range = ((-.5, .5), (-.5, .5), (-.5, .5))
        # self.phase_range = ((.0, np.pi), (.0, np.pi), (.0, np.pi))
        # self.n_components = (1, 1, 1)
        self.n_sines = 5
        self.periods = np.array([.5, 1., .5])[:self.n_sines]
        self.seq_length = seq_length
        self.n_periods = n_periods
        self.n_batch = n_batch
        self.random_state = np.random.RandomState(seed=sine_seed)
        self.seed = sine_seed
        self.static = static
        self.dt_step = dt_step

    @property
    def shape(self):
        return (self.n_batch, self.seq_length), (self.n_batch, self.seq_length), (self.n_batch, self.seq_length),\
               (self.n_batch, self.seq_length)

    def __call__(self):
        t = np.linspace(0, 1 * 2 * np.pi, self.seq_length // self.n_periods)
        while True:
            if self.static:
                self.random_state = np.random.RandomState(seed=self.seed)
            periods = self.random_state.rand(self.n_batch, self.n_sines) * .7 + .3
            phase_motor0 = self.random_state.rand(self.n_batch, self.n_sines) * np.pi * 2
            amp0 = self.random_state.rand(self.n_batch, self.n_sines) * 1.5 * 20
            omega0 = np.sin(t[..., None, None] / periods[None, ...] + phase_motor0[None, ...]) * amp0[None, ...]
            # omega0 = (omega0 / (omega0.max(0) - omega0.min(0)) * 1.).sum(-1)
            omega0 = omega0.sum(-1)
            # phi0 = np.clip(dt_step * np.cumsum(omega0, 0), -np.pi / 2, 0)
            phi0 = self.dt_step * np.cumsum(omega0, 0)
            phi0_max = np.max(phi0, 0)
            phi0_min = np.min(phi0, 0)
            # assert np.allclose(phi0_max, -phi0_min)
            selector = np.logical_or(phi0_max > np.pi / 2, phi0_min < -np.pi / 2)
            sc = (np.pi / 2) / phi0_max[selector]
            sc2 = (-np.pi / 2) / phi0_min[selector]
            sc[sc < 0.] = 1.
            sc2[sc2 < 0.] = 1.
            sc = np.min((sc, sc2), 0)
            phi0[:, selector] = sc[None, :] * phi0[:, selector]
            omega0[:, selector] = sc[None, :] * omega0[:, selector]
            # fig, ax = plt.subplots(1, figsize=(6, 5))
            # ax.plot(phi0)
            # fig.savefig(os.path.expanduser('~/tempfig.png'), dpi=200)
            assert np.all(np.abs(phi0) - 1e-5 <= np.pi / 2)
            # phi0_max = np.max(phi0)
            # if phi0_max > 0:
            #     phi0 -= phi0_max

            phase_motor1 = self.random_state.rand(self.n_batch, self.n_sines) * np.pi * 2
            amp1 = self.random_state.rand(self.n_batch, self.n_sines) * 1.5
            periods = self.random_state.rand(self.n_batch, self.n_sines) * .7 + .3
            omega1 = np.sin(t[..., None, None] / periods[None, ...] + phase_motor1[None, ...]) * amp1[None, ...]
            omega1 = (omega1 / (omega1.max(0) - omega1.min(0)) * 20.).sum(-1)
            # phi1 = phi0 + np.clip(dt_step * np.cumsum(omega1, 0), 0, np.pi / 2) + np.pi / 2
            phi1_rel = self.dt_step * np.cumsum(omega1, 0)
            # phi1_rel_min = np.min(phi1_rel)
            # if phi1_rel_min < 0:
            #     phi1 = phi0 + phi1_rel - phi1_rel_min
            phi1_max = np.max(phi1_rel, 0)
            phi1_min = np.min(phi1_rel, 0)
            selector = np.logical_or(phi1_max > np.pi / 2, phi1_min < -np.pi / 2)
            sc = (np.pi / 2) / phi1_max[selector]
            sc2 = (-np.pi / 2) / phi1_min[selector]
            sc[sc < 0.] = 1.
            sc2[sc2 < 0.] = 1.
            sc = np.min((sc, sc2), 0)
            # sc = np.min(((np.pi / 4) / phi1_max[selector], (-np.pi / 4) / phi1_min[selector]), 0)
            phi1_rel[:, selector] = sc[None, :] * phi1_rel[:, selector]
            assert np.all(np.abs(phi1_rel) - 1e-5 <= np.pi / 2)
            omega1[:, selector] = sc[None, :] * omega1[:, selector]
            phi1 = phi0 + phi1_rel + np.pi / 2

            # x0, x1 = amp0[None, ...] * np.cos(phi0), amp1[None, ...] * np.cos(phi1)
            # y0, y1 = amp0[None, ...] * np.sin(phi0), amp1[None, ...] * np.sin(phi1)

            x = (np.cos(phi0) + np.cos(phi1)).T * .5
            y = (np.sin(phi0) + np.sin(phi1)).T * .5
            # fig, ax = plt.subplots(4, 4, figsize=(6, 5))
            # print(x[0, 0])
            # print(y[0, 0])
            # print(x[1, 0])
            # print(y[1, 0])
            # for i in range(4):
            #     for j in range(4):
            #         ax[i][j].scatter(x[i * 4 + j, :FLAGS.plasticity_every:10], y[i * 4 + j, :FLAGS.plasticity_every:10],
            #                          s=1, c=t[:FLAGS.plasticity_every:10], cmap='coolwarm')
            #         ax[i][j].plot([x[0, 0]], [y[0, 0]], '.', color='C0')
            #         ax[i][j].set_ylim([-1, 1])
            #         ax[i][j].set_xlim([-1, 1])
            #         ax[i][j].xaxis.set_ticks([])
            #         ax[i][j].yaxis.set_ticks([])
            # fig.savefig(os.path.expanduser('~/trajectory.png'), dpi=200)
            # fig.savefig(os.path.expanduser('~/trajectory.svg'),)
            # quit()
            # phase_x = np.random.rand(self.n_batch, self.n_sines) * 2 * np.pi
            # phase_y = np.random.rand(self.n_batch, self.n_sines) * 2 * np.pi
            # amp = np.random.rand(self.n_batch, self.n_sines) * .8
            # x = amp[None, ...] * np.cos(t[..., None, None] / self.periods[None, ...] + phase_x[None, ...])
            # y = amp[None, ...] * np.sin(t[..., None, None] / self.periods[None, ...] + phase_y[None, ...])
            # x = np.sum(x, -1).T
            # y = np.sum(y, -1).T
            # x = np.cumsum(x, 1) * dt_step
            # y = np.cumsum(y, 1) * dt_step
            x = np.tile(x[:, :], (1, 2))
            y = np.tile(y[:, :], (1, 2))
            omega0 = np.tile(omega0.T[:, :], (1, 2))
            omega1 = np.tile(omega1.T[:, :], (1, 2))
            yield x, y, omega0, omega1
