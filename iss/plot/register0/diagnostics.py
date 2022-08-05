from ..stitch.diagnostics import shift_info_plot
from ...setup import Notebook


def view_reg_shift_info(nb: Notebook):
    """
    For all shifts to imaging rounds from the reference round computed in the `register_initial` section
    of the pipeline, this plots the values of the shifts found and the `score` compared to
    the `score_thresh`.

    For each round, there will be 3 plots:

    * y shift vs x shift for all tiles
    * z shift vs x shift for all tiles
    * `score` vs `score_thresh` for all tiles (a green score = score_thresh line is plotted in this).

    In each case, the markers in the plots are numbers.
    These numbers indicate the tile the shift was found for.
    The number will be blue if `score > score_thresh` and red otherwise.

    Args:
        nb: Notebook containing at least the `register_initial` page.
    """
    shift_info = {}
    if nb.basic_info.is_3d:
        ndim = 3
    else:
        ndim = 2
    for r in nb.basic_info.use_rounds:
        name = f'Round {r}'
        shift_info[name] = {}
        shift_info[name]['shift'] = nb.register_initial_debug.shift[nb.basic_info.use_tiles, r, :ndim]
        shift_info[name]['tile'] = nb.basic_info.use_tiles
        shift_info[name]['score'] = nb.register_initial_debug.shift_score[nb.basic_info.use_tiles, r]
        shift_info[name]['score_thresh'] = nb.register_initial_debug.shift_score_thresh[nb.basic_info.use_tiles, r]
    shift_info_plot(shift_info, f"Shifts from round {nb.basic_info.ref_round}, channel {nb.basic_info.ref_channel} "
                                f"to channel {nb.register_initial_debug.shift_channel} for each round and tile")
