from typing import Optional
import numpy as np
from ..stitch import view_shifts
from ..stitch.diagnostics import shift_info_plot
from ...find_spots import spot_yxz
from ...setup import Notebook
from ...stitch import compute_shift


def view_register_shift_info(nb: Notebook, outlier: bool = False):
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
        outlier: If `True`, will plot `nb.register_initial.shift_outlier` instead of
            `nb.register_initial.shift`. In this case, only tiles for which
            the two are different are plotted for each round.
    """
    shift_info = {}
    if nb.basic_info.is_3d:
        ndim = 3
    else:
        ndim = 2
    for r in nb.basic_info.use_rounds:
        name = f'Round {r}'
        shift_info[name] = {}
        shift_info[name]['tile'] = nb.basic_info.use_tiles
        if outlier:
            shift_info[name]['shift'] = nb.register_initial.shift_outlier[nb.basic_info.use_tiles, r, :ndim]
            shift_info[name]['score'] = nb.register_initial.shift_score_outlier[nb.basic_info.use_tiles, r]
        else:
            shift_info[name]['shift'] = nb.register_initial.shift[nb.basic_info.use_tiles, r, :ndim]
            shift_info[name]['score'] = nb.register_initial.shift_score[nb.basic_info.use_tiles, r]
        shift_info[name]['score_thresh'] = nb.register_initial.shift_score_thresh[nb.basic_info.use_tiles, r]

    if outlier:
        title_start = "Outlier "
    else:
        title_start = ""
    shift_info_plot(shift_info, f"{title_start}Shifts found in register_initial part of pipeline "
                                f"from round {nb.basic_info.ref_round}, channel "
                                f"{nb.basic_info.ref_channel} to channel "
                                f"{nb.register_initial.shift_channel} for each round and tile")


def view_register_search(nb: Notebook, t: int, r: int, c: Optional[int] = None,
                         return_shift: bool = False) -> Optional[np.ndarray]:
    """
    Function to plot results of exhaustive search to find shift between `ref_round/ref_channel` and
    round `r`, channel `c` for tile `t`. This shift will then be used as the starting point when running point cloud
    registration to find affine transform.
    Useful for debugging the `register_initial` section of the pipeline.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
        t: tile interested in.
        r: Want to find the shift between the reference round and this round.
        c: Want to find the shift between the reference channel and this channel. If `None`, `config['shift_channel']`
            will be used, as it is in the pipeline.
        return_shift: If True, will return shift found and will not call plt.show() otherwise will return None.

    Returns:
        `best_shift` - `float [shift_y, shift_x, shift_z]`.
            Best shift found. `shift_z` is in units of z-pixels.
    """
    config = nb.get_config()['register_initial']
    if c is None:
        c = config['shift_channel']
        if c is None:
            c = nb.basic_info.ref_channel
    if not np.isin(c, nb.basic_info.use_channels):
        raise ValueError(f"c should be in nb.basic_info.use_channels, but value given is\n"
                         f"{c} which is not in use_channels = {nb.basic_info.use_channels}.")

    coords = ['y', 'x', 'z']
    shifts = [{}]
    for i in range(len(coords)):
        shifts[0][coords[i]] = np.arange(config['shift_min'][i],
                                         config['shift_max'][i] +
                                         config['shift_step'][i] / 2, config['shift_step'][i]).astype(int)
    if not nb.basic_info.is_3d:
        config['nz_collapse'] = None
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0
        shifts[0]['z'] = np.array([0], dtype=int)
    shifts = shifts * nb.basic_info.n_rounds  # get one set of shifts for each round
    c_ref = nb.basic_info.ref_channel
    r_ref = nb.basic_info.ref_round
    # to convert z coordinate units to xy pixels when calculating distance to nearest neighbours
    z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
    print(f'Finding shift between round {r_ref}, channel {c_ref} to round {r}, channel {c} for tile {t}')
    shift, shift_score, shift_score_thresh, debug_info = \
        compute_shift(spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref),
                      spot_yxz(nb.find_spots.spot_details, t, r, c),
                      config['shift_score_thresh'], config['shift_score_thresh_multiplier'],
                      config['shift_score_thresh_min_dist'], config['shift_score_thresh_max_dist'],
                      config['neighb_dist_thresh'], shifts[r]['y'], shifts[r]['x'], shifts[r]['z'],
                      config['shift_widen'], config['shift_max_range'], z_scale,
                      config['nz_collapse'], config['shift_step'][2])
    title = f'Shift between r={r_ref}, c={c_ref} and r={r}, c={c} for tile {t}. YXZ Shift = {shift}.'
    if return_shift:
        show = False
    else:
        show = True
    view_shifts(debug_info['shifts_2d'], debug_info['scores_2d'], debug_info['shifts_3d'],
                debug_info['scores_3d'], shift, debug_info['min_score_2d'], debug_info['shift_2d_initial'],
                shift_score_thresh, debug_info['shift_thresh'], config['shift_score_thresh_min_dist'],
                config['shift_score_thresh_max_dist'], title, show)
    if return_shift:
        return shift
