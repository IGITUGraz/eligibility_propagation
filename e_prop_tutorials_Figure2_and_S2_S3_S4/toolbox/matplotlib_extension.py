__author__ = 'guillaume'

import numpy as np


def raster_plot(ax, spikes, linewidth=0.8, **kwargs):
    n_t, n_n = spikes.shape
    event_times, event_ids = np.where(spikes)
    max_spike = 10000
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]

    for n, t in zip(event_ids, event_times):
        ax.vlines(t, n + 0., n + 1., linewidth=linewidth, **kwargs)

    ax.set_ylim([0 + .5, n_n + .5])
    ax.set_xlim([0, n_t])
    ax.set_yticks([0, n_n])


def strip_right_top_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def arrow_trajectory(ax,data,epsi=0,hdw=.03,lab='',fact=.8,color=(1.,1.,1.,1.),arrow_tick_steps=[],**kwargs):

    fc = tuple(np.clip(np.array(color) * fact,0,1.))

    ploted_lab = False

    X = data[:-1,:]
    dX = data[1:,:] - data[:-1,:]

    t0 = 0
    T = data.shape[0]-1

    if epsi > 0:
        while sum(dX[T-1]**2) / np.mean( np.sum(dX**2,axis=1)) < epsi: T = T-1
        while sum(dX[t0]**2) / np.mean(np.sum(dX**2,axis=1)) < epsi: t0 = t0+1

    ax.scatter(data[t0,0],data[t0,1],s=50,facecolor=fc,color=color,**kwargs)


    for t in np.arange(t0+1,T):
        x,y = X[t-1,:]
        dx,dy = dX[t-1,:]

        if t == T-1:
            headwidth = hdw
            head_length = hdw * 1.5
        elif t in arrow_tick_steps:
            headwidth = hdw
            head_length = hdw * 0.15
        else:
            headwidth = 0.
            head_length = 0.

        if dx != 0 or dy != 0:
            if ploted_lab:
                p = patches.FancyArrow(x, y, dx, dy,facecolor=color,edgecolor=fc,head_width=headwidth,head_length=head_length,**kwargs)
            else:
                ploted_lab = True
                p = patches.FancyArrow(x, y, dx, dy,facecolor=color,edgecolor=fc,head_width=headwidth,head_length=head_length,label=lab,**kwargs)
            ax.add_patch(p)