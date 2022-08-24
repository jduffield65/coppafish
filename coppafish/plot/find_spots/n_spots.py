from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from ...setup import Notebook


def n_spots_grid(nb: Notebook, n_spots_thresh: Optional[int] = None):
    """
    Plots a grid indicating the number of spots detected on each tile, round and channel.

    Args:
        nb: *Notebook* containing `find_spots` page.
        n_spots_thresh: tiles/rounds/channels with fewer spots than this will be highlighted.
            If `None`, will use `n_spots_warn_fraction` from config file.
    """
    if n_spots_thresh is None:
        config = nb.get_config()['find_spots']
        if nb.basic_info.is_3d:
            n_spots_thresh = config['n_spots_warn_fraction'] * config['max_spots_3d'] * nb.basic_info.nz
        else:
            n_spots_thresh = config['n_spots_warn_fraction'] * config['max_spots_2d']
        n_spots_thresh = int(np.ceil(n_spots_thresh))
    use_tiles = np.asarray(nb.basic_info.use_tiles)
    use_rounds = np.asarray(nb.basic_info.use_rounds)  # don't consider anchor in this analysis
    if len(use_rounds) > 0:
        use_channels = np.asarray(nb.basic_info.use_channels)
        spot_no = nb.find_spots.spot_no[np.ix_(use_tiles, use_rounds, use_channels)]
        spot_no = np.moveaxis(spot_no, 1, 2)  # put rounds last
        spot_no = spot_no.reshape(len(use_tiles), -1).T  # spot_no[n_rounds, t] is now spot_no[t, r=0, c=1]
        n_round_channels = len(use_rounds) * len(use_channels)
        y_labels = np.tile(use_rounds, len(use_channels))
        vmax = spot_no.max()  # clip colorbar at max of imaging rounds/channels because anchor can be a lot higher
    else:
        # Deal with case where only anchor round
        use_channels = np.asarray([])
        spot_no = np.zeros((0, len(use_tiles)), dtype=np.int32)
        n_round_channels = 0
        y_labels = np.zeros(0, dtype=int)
        vmax = None
    if nb.basic_info.use_anchor:
        anchor_spot_no = nb.find_spots.spot_no[use_tiles, nb.basic_info.anchor_round,
                                               nb.basic_info.anchor_channel][np.newaxis]
        spot_no = np.append(spot_no, anchor_spot_no, axis=0)
        y_labels = y_labels.astype(str).tolist()
        y_labels += ['Anchor']
        n_round_channels += 1
        if vmax is None:
            vmax = spot_no.max()
    fig, ax = plt.subplots(1, 1, figsize=(np.clip(5+len(use_tiles)/2, 3, 18), 12))
    subplot_adjust = [0.12, 1, 0.07, 0.9]
    fig.subplots_adjust(left=subplot_adjust[0], right=subplot_adjust[1],
                        bottom=subplot_adjust[2], top=subplot_adjust[3])
    im = ax.imshow(spot_no, aspect='auto', vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_yticks(np.arange(n_round_channels))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(len(use_tiles)))
    ax.set_xticklabels(use_tiles)
    ax.set_xlabel('Tile')

    for c in range(len(use_channels)):
        y_ind = 1 - c * 1/(len(use_channels))
        ax.text(-0.1, y_ind, f"Channel {use_channels[c]}", va="top", ha="left", transform=ax.transAxes,
                rotation='vertical')
    fig.supylabel('Channel/Round', transform=ax.transAxes, x=-0.15)
    plt.xticks(rotation=90)
    low_spots = np.where(spot_no<n_spots_thresh)
    for j in range(len(low_spots[0])):
        rectangle = plt.Rectangle((low_spots[1][j] - 0.5, low_spots[0][j] - 0.5), 1, 1,
                                  fill=False, ec="r", linestyle=':', lw=2)
        ax.add_patch(rectangle)
    plt.suptitle(f"Number of Spots Found on each Tile, Round and Channel")
    plt.show()
