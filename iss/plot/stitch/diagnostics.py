import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from ...setup import Notebook


def shift_info_plot(shift_info: dict[dict], title: Optional[str] = None):
    """

    Args:
        shift_info:
        title:

    Returns:

    """
    n_cols = len(shift_info)
    col_titles = list(shift_info.keys())
    n_rows = len(shift_info[col_titles[0]]['shift'][0])  # 2 if 2D shift, 3 if 3D.
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 7))
    fig.subplots_adjust(hspace=0.4, bottom=0.08, left=0.06, right=0.97, top=0.9)
    if n_cols == 1:
        ax = ax[:, np.newaxis]
    for i in range(n_cols):
        shift_info_i = shift_info[col_titles[i]]
        # good tiles are blue, bad tiles are red
        n_tiles = len(shift_info[col_titles[i]]['tile'])
        tile_color = np.full(n_tiles, 'b')
        tile_color[(shift_info_i['score'] < shift_info_i['score_thresh']).flatten()] = 'r'
        for t in range(n_tiles):
            ax[0, i].text(shift_info_i['shift'][t, 1], shift_info_i['shift'][t, 0],
                          str(shift_info_i['tile'][t]), color=tile_color[t], fontsize=12,
                          ha='center', va='center')
        ax[0, i].set_xlim([np.min(shift_info_i['shift'][:, 1]) - 3, np.max(shift_info_i['shift'][:, 1]) + 3])
        ax[0, i].set_ylim([np.min(shift_info_i['shift'][:, 0]) - 3, np.max(shift_info_i['shift'][:, 0]) + 3])
        if i == int(np.ceil(n_cols/2)-1):
            ax[0, i].set_xlabel('X Shift')
        if i == 0:
            ax[0, i].set_ylabel('Y Shift')
        ax[0, i].set_title(col_titles[i])

        row_ind = 1
        if n_rows == 3:
            for t in range(n_tiles):
                ax[row_ind, i].text(shift_info_i['shift'][t, 1], shift_info_i['shift'][t, 2],
                                    str(shift_info_i['tile'][t]), color=tile_color[t], fontsize=12,
                                    ha='center', va='center')
            ax[row_ind, i].set_xlim([np.min(shift_info_i['shift'][:, 1]) - 3, np.max(shift_info_i['shift'][:, 1]) + 3])
            ax[row_ind, i].set_ylim([np.min(shift_info_i['shift'][:, 2]) - 1, np.max(shift_info_i['shift'][:, 2]) + 1])
            if i == int(np.ceil(n_cols/2)-1):
                ax[row_ind, i].set_xlabel('X Shift')
            if i == 0:
                ax[row_ind, i].set_ylabel('Z Shift')
            row_ind += 1

        # Plot line so that all with score > score_thresh are above it
        score_min = np.min(np.vstack([shift_info_i['score_thresh'], shift_info_i['score']])) - 5
        score_max = np.max(np.vstack([shift_info_i['score_thresh'], shift_info_i['score']])) + 5
        ax[row_ind, i].set_xlim([score_min, score_max])
        ax[row_ind, i].set_ylim([score_min, score_max])
        ax[row_ind, i].plot([score_min, score_max], [score_min, score_max], 'lime', linestyle=':', linewidth=2,
                            alpha=0.5)
        for t in range(n_tiles):
            ax[row_ind, i].text(shift_info_i['score_thresh'][t], shift_info_i['score'][t],
                                str(shift_info_i['tile'][t]), color=tile_color[t], fontsize=12,
                                ha='center', va='center')
        if i == int(np.ceil(n_cols/2)-1):
            ax[row_ind, i].set_xlabel('Score Threshold')
        if i == 0:
            ax[row_ind, i].set_ylabel('Score')
    if title is not None:
        plt.suptitle(title)
    plt.show()


def view_stitch_shift_info(nb: Notebook):
    """

    Args:
        nb:

    Returns:

    """
    if nb.basic_info.is_3d:
        ndim = 3
    else:
        ndim = 2
    shift_info = {}
    if len(nb.stitch.south_shifts) > 0:
        shift_info['South'] = {}
        shift_info['South']['shift'] = nb.stitch.south_shifts[:, :ndim]
        shift_info['South']['tile'] = nb.stitch.south_pairs[:, 0]
        shift_info['South']['score'] = nb.stitch.south_score
        shift_info['South']['score_thresh'] = nb.stitch.south_score_thresh
    if len(nb.stitch.west_shifts) > 0:
        shift_info['West'] = {}
        shift_info['West']['shift'] = nb.stitch.west_shifts[:, :ndim]
        shift_info['West']['tile'] = nb.stitch.west_pairs[:, 0]
        shift_info['West']['score'] = nb.stitch.west_score
        shift_info['West']['score_thresh'] = nb.stitch.west_score_thresh
    shift_info_plot(shift_info, "Shifts found in stitch part of pipeline between each tile and the neighbouring tile "
                                "in the direction specified")
