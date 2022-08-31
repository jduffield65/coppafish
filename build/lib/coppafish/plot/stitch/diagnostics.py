import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from ...setup import Notebook
plt.style.use('dark_background')


def shift_info_plot(shift_info: dict, title: Optional[str] = None, score_plot_thresh: int = 0,
                    fig: Optional[plt.Figure] = None, ax: Optional[np.ndarray] = None, return_ax: bool = False):
    """
    If `shift_info` contains $n$ keys, this will produce an $n$ column x 3 row grid of subplots.
    For each key in `shift_info` dictionary, there are 3 plots:

    * y shift vs x shift
    * z shift vs x shift
    * `score` vs `score_thresh` or `n_matches` vs `error`

    In each case, the markers in the plots are numbers.
    These numbers are given by `shift_info[key][tile]`.
    The number will be blue if `score > score_thresh` and red otherwise.

    Args:
        shift_info: Dictionary containing $n$ dictionaries.
            Each of these dictionaries contains (either (`score` and `score_thresh`) or (`n_matches`,
            `n_matches_thresh` and `error`)):

            * shift - `float [n_tiles x 3]`. $yxz$ shifts for each tile.
            * tile - `int [n_tiles]`. Indicates tile each shift was found for.
            * score - `float [n_tiles]`. Indicates `score` found for each shift (approx number of matches between
                point clouds).
            * score_thresh - `float [n_tiles]`. If `score<score_thresh`, it will be shown in red.
            * n_matches - `int [n_tiles]`. Indicates `score` found for each shift (approx number of matches between
                point clouds).
            * n_matches_thresh - `int [n_tiles]`. If `n_matches<n_matches_thresh`, it will be shown in red.
            * error - `float [n_tiles]`. Average distance between neighbours.
            * x_lim - `float [n_plots x 2]`. Can optionally specify number axis limits for each plot.
            * y_lim - `float [n_plots x 2]`. Can optionally specify number axis limits for each plot.
        title: Overall title for the plot.
        score_plot_thresh: Only shifts with `score` (or `n_matches`) > `score_plot_thresh` are shown.
        fig: Can provide previous figure to plot on.
        ax: Can provide array of plt.Axes to plot on.
        return_ax: If `True`, ax will be returned and `plt.show()` will not be run.
    """
    n_cols = len(shift_info)
    col_titles = list(shift_info.keys())
    n_rows = len(shift_info[col_titles[0]]['shift'][0])  # 2 if 2D shift, 3 if 3D.
    if fig is None:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 7))
        fig.subplots_adjust(hspace=0.4, bottom=0.08, left=0.06, right=0.97, top=0.9)
    if n_cols == 1 and len(ax.shape) == 1:
        ax = ax[:, np.newaxis]
    for i in range(n_cols):
        shift_info_i = shift_info[col_titles[i]]
        # good tiles are blue, bad tiles are red
        n_tiles = len(shift_info[col_titles[i]]['tile'])
        tile_color = np.full(n_tiles, 'b')
        if 'score' in shift_info_i:
            tile_color[(shift_info_i['score'] < shift_info_i['score_thresh']).flatten()] = 'r'
            skip_tile = shift_info_i['score'] <= score_plot_thresh  # don't plot if score = 0
        elif 'n_matches' in shift_info_i:
            tile_color[(shift_info_i['n_matches'] < shift_info_i['n_matches_thresh']).flatten()] = 'r'
            skip_tile = shift_info_i['n_matches'] <= score_plot_thresh  # don't plot if n_matches = 0
        else:
            raise ValueError(f"shift_info must contain either a 'score' or 'n_matches' key")
        for t in range(n_tiles):
            if skip_tile[t]:
                continue
            ax[0, i].text(shift_info_i['shift'][t, 1], shift_info_i['shift'][t, 0],
                          str(shift_info_i['tile'][t]), color=tile_color[t], fontsize=12,
                          ha='center', va='center')
        if 'x_lim' in shift_info_i:
            ax[0, i].set_xlim(shift_info_i['x_lim'][0])
        else:
            ax[0, i].set_xlim([np.min(shift_info_i['shift'][:, 1]) - 3, np.max(shift_info_i['shift'][:, 1]) + 3])
        if 'y_lim' in shift_info_i:
            ax[0, i].set_ylim(shift_info_i['y_lim'][0])
        else:
            ax[0, i].set_ylim([np.min(shift_info_i['shift'][:, 0]) - 3, np.max(shift_info_i['shift'][:, 0]) + 3])
        if i == int(np.ceil(n_cols/2)-1):
            ax[0, i].set_xlabel('X Shift')
        if i == 0:
            ax[0, i].set_ylabel('Y Shift')
        ax[0, i].set_title(col_titles[i])

        row_ind = 1
        if n_rows == 3:
            for t in range(n_tiles):
                if skip_tile[t]:
                    continue
                ax[row_ind, i].text(shift_info_i['shift'][t, 1], shift_info_i['shift'][t, 2],
                                    str(shift_info_i['tile'][t]), color=tile_color[t], fontsize=12,
                                    ha='center', va='center')
            if 'x_lim' in shift_info_i:
                ax[row_ind, i].set_xlim(shift_info_i['x_lim'][row_ind])
            else:
                ax[row_ind, i].set_xlim([np.min(shift_info_i['shift'][:, 1]) - 3,
                                         np.max(shift_info_i['shift'][:, 1]) + 3])
            if 'y_lim' in shift_info_i:
                ax[row_ind, i].set_ylim(shift_info_i['y_lim'][row_ind])
            else:
                ax[row_ind, i].set_ylim([np.min(shift_info_i['shift'][:, 2]) - 1,
                                         np.max(shift_info_i['shift'][:, 2]) + 1])
            if i == int(np.ceil(n_cols/2)-1):
                ax[row_ind, i].set_xlabel('X Shift')
            if i == 0:
                ax[row_ind, i].set_ylabel('Z Shift')
            row_ind += 1

        if 'score' in shift_info_i:
            # Plot line so that all with score > score_thresh are above it
            if 'x_lim' in shift_info_i and 'y_lim' in shift_info_i:
                score_min = np.min(shift_info_i['y_lim'][row_ind, 0], shift_info_i['x_lim'][row_ind, 0]) - 5
                score_max = np.max(shift_info_i['y_lim'][row_ind, 1], shift_info_i['x_lim'][row_ind, 1]) + 5
            else:
                score_min = np.min(np.vstack([shift_info_i['score_thresh'], shift_info_i['score']])) - 5
                score_max = np.max(np.vstack([shift_info_i['score_thresh'], shift_info_i['score']])) + 5
            ax[row_ind, i].set_xlim([score_min, score_max])
            ax[row_ind, i].set_ylim([score_min, score_max])
            ax[row_ind, i].plot([score_min, score_max], [score_min, score_max], 'lime', linestyle=':', linewidth=2,
                                alpha=0.5)
            for t in range(n_tiles):
                if skip_tile[t]:
                    continue
                ax[row_ind, i].text(shift_info_i['score_thresh'][t], shift_info_i['score'][t],
                                    str(shift_info_i['tile'][t]), color=tile_color[t], fontsize=12,
                                    ha='center', va='center')
            if i == int(np.ceil(n_cols/2)-1):
                ax[row_ind, i].set_xlabel('Score Threshold')
            if i == 0:
                ax[row_ind, i].set_ylabel('Score')
        elif 'n_matches' in shift_info_i:
            if 'x_lim' in shift_info_i:
                ax[row_ind, i].set_xlim(shift_info_i['x_lim'][row_ind])
            else:
                ax[row_ind, i].set_xlim([np.min(shift_info_i['error']) - 0.1, np.max(shift_info_i['error']) + 0.1])
            if 'y_lim' in shift_info_i:
                ax[row_ind, i].set_ylim(shift_info_i['y_lim'][row_ind])
            else:
                ax[row_ind, i].set_ylim([np.min(shift_info_i['n_matches']) - 100,
                                         np.max(shift_info_i['n_matches']) + 100])
            for t in range(n_tiles):
                if skip_tile[t]:
                    continue
                ax[row_ind, i].text(shift_info_i['error'][t], shift_info_i['n_matches'][t],
                                    str(shift_info_i['tile'][t]), color=tile_color[t], fontsize=12,
                                    ha='center', va='center')
            if i == int(np.ceil(n_cols/2)-1):
                ax[row_ind, i].set_xlabel('Error')
            if i == 0:
                ax[row_ind, i].set_ylabel(r'$n_{matches}$')
    if title is not None:
        plt.suptitle(title)
    if return_ax:
        return ax
    else:
        plt.show()


def view_stitch_shift_info(nb: Notebook, outlier: bool = False):
    """
    For all north/south and east/west shifts computed in the `stitch` section
    of the pipeline, this plots the values of the shifts found and the `score` compared to
    the `score_thresh`.

    For each direction, there will be 3 plots:

    * y shift vs x shift for all pairs of neighbouring tiles
    * z shift vs x shift for all pairs of neighbouring tiles
    * `score` vs `score_thresh` for all pairs of neighbouring tiles
    (a green score = score_thresh line is plotted in this).

    In each case, the markers in the plots are numbers.
    These numbers indicate the tile, the shift was applied to,
    to take it to its north or east neighbour i.e. `nb.stitch.south_pairs[:, 0]`
    or `nb.stitch.west_pairs[:, 0]`.
    The number will be blue if `score > score_thresh` and red otherwise.

    Args:
        nb: Notebook containing at least the `stitch` page.
        outlier: If `True`, will plot `nb.stitch.south_shift_outlier` instead of
            `nb.stitch.south_shift`. In this case, only tiles for which
            the two are different are plotted for each round.
    """
    if nb.basic_info.is_3d:
        ndim = 3
    else:
        ndim = 2
    shift_info = {}
    if len(nb.stitch.south_shifts) > 0:
        shift_info['South'] = {}
        shift_info['South']['tile'] = nb.stitch.south_pairs[:, 0]
        shift_info['South']['score_thresh'] = nb.stitch.south_score_thresh
        if outlier:
            shift_info['South']['shift'] = nb.stitch.south_outlier_shifts[:, :ndim]
            shift_info['South']['score'] = nb.stitch.south_outlier_score
        else:
            shift_info['South']['shift'] = nb.stitch.south_shifts[:, :ndim]
            shift_info['South']['score'] = nb.stitch.south_score
    if len(nb.stitch.west_shifts) > 0:
        shift_info['West'] = {}
        shift_info['West']['tile'] = nb.stitch.west_pairs[:, 0]
        shift_info['West']['score_thresh'] = nb.stitch.west_score_thresh
        if outlier:
            shift_info['West']['shift'] = nb.stitch.west_outlier_shifts[:, :ndim]
            shift_info['West']['score'] = nb.stitch.west_outlier_score
        else:
            shift_info['West']['shift'] = nb.stitch.west_shifts[:, :ndim]
            shift_info['West']['score'] = nb.stitch.west_score
    if outlier:
        title_start = "Outlier "
    else:
        title_start = ""
    shift_info_plot(shift_info, f"{title_start}Shifts found in stitch part of pipeline between each tile and the "
                                f"neighbouring tile in the direction specified")
