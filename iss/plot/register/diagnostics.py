import numpy as np
import distinctipy
import matplotlib.pyplot as plt
import warnings
from ...setup import Notebook
plt.style.use('dark_background')


def scale_box_plots(nb: Notebook):
    """
    Function to plot distribution of chromatic aberration scaling amongst tiles for each round and channel.
    Want very similar values for a given channel across all tiles and rounds for each dimension.
    Also expect $y$ and $x$ scaling to be very similar. $z$ scaling different due to unit conversion.

    Args:
        nb: *Notebook* containing the `register` and `register_debug` *NotebookPages*.
    """
    if nb.basic_info.is_3d:
        ndim = 3
        if np.ptp(nb.register.transform[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                               nb.basic_info.use_channels)][:, :, :, 2, 2]) < 1e5:
            ndim = 2
            warnings.warn("Not showing z-scaling as all are the same")
    else:
        ndim = 2

    fig, ax = plt.subplots(ndim, figsize=(10, 6), sharex=True)
    ax[0].get_shared_y_axes().join(ax[0], ax[1])
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15)
    y_titles = ["Scaling - Y", "Scaling - X", "Scaling - Z"]
    n_use_channels = len(nb.basic_info.use_channels)
    # different boxplot color for each channel
    # Must be distinct from black and white
    channel_colors = distinctipy.get_colors(n_use_channels, [(0, 0, 0), (1, 1, 1)])
    for i in range(ndim):
        box_data = [nb.register.transform[nb.basic_info.use_tiles, r, c, i, i] for c in nb.basic_info.use_channels
                    for r in nb.basic_info.use_rounds]
        bp = ax[i].boxplot(box_data, notch=0, sym='+', patch_artist=True)
        leg_markers = []
        c = -1
        for j in range(len(box_data)):
            if j % n_use_channels == 0:
                c += 1
                leg_markers = leg_markers + [bp['boxes'][j]]
            bp['boxes'][j].set_facecolor(channel_colors[c])
        ax[i].set_ylabel(y_titles[i])

        if i == ndim-1:
            tick_labels = np.tile(nb.basic_info.use_rounds, n_use_channels).tolist()
            leg_labels = nb.basic_info.use_channels
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels)
            ax[i].legend(leg_markers, leg_labels, title='Channel')
            ax[i].set_xlabel('Round')
    ax[0].set_title('Boxplots showing distribution of scalings due to\nchromatic aberration amongst tiles for each '
                    'round and channel')
    plt.show()
