import numpy as np
from ..stitch import compute_shift
from ..find_spots import spot_yxz
from .stitch import view_shifts
from ..setup import Notebook
from typing import Optional


def view_initial_shift(nb: Notebook, config: dict, t: int, r: int, c: Optional[int] = None):
    """
    Function to plot results of exhaustive search to find shift between `ref_round/ref_channel` and
    round `r`, channel `c` for tile `t`. This shift will then be used as the starting point when running point cloud
    registration to find affine transform.
    Useful for debugging the `register_initial` section of the pipeline.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
        config: Dictionary obtained from `'register_initial'` section of config file.
        t: tile interested in.
        r: Want to find the shift between the reference round and this round.
        c: Want to find the shift between the reference channel and this channel. If `None`, `config['shift_channel']`
            will be used, as it is in the pipeline.
    """
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
    view_shifts(debug_info['shifts_2d'], debug_info['scores_2d'], debug_info['shifts_3d'],
                debug_info['scores_3d'], shift, shift_score_thresh, title)
