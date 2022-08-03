import numpy as np
from ..stitch import compute_shift
from ..find_spots import spot_yxz, get_isolated_points
from ..pcr import get_single_affine_transform
from ..spot_colors.base import apply_transform
from .stitch import view_shifts, view_point_clouds
from ..setup import Notebook
from typing import Optional


def view_initial_shift(nb: Notebook, t: int, r: int, c: Optional[int] = None,
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
                debug_info['scores_3d'], shift, shift_score_thresh, title, show)
    if return_shift:
        return shift


def view_pcr(nb: Notebook, t: int, r: int, c: int):
    """
    Function to plot results of point cloud registration to find affine transform between `ref_round/ref_channel` and
    round `r`, channel `c` for tile `t`.
    Useful for debugging the `register` section of the pipeline.

    Args:
        nb: Notebook containing results of the experiment. Must contain `find_spots` page.
            If contains register_initial_debug and/or register pages, then transform from these will be used.
        t: tile interested in.
        r: Want to find the transform between the reference round and this round.
        c: Want to find the transform between the reference channel and this channel.
    """
    config = nb.get_config()
    if nb.basic_info.is_3d:
        neighb_dist_thresh = config['register']['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['register']['neighb_dist_thresh_2d']
    z_scale = [1, 1, nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy]
    point_clouds = []
    # 1st point cloud is imaging one as does not change
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r, c)]
    r_ref = nb.basic_info.ref_round
    c_ref = nb.basic_info.ref_channel
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref)]
    for i in range(2):
        # only keep isolated spots, those whose second neighbour is far away
        isolated = get_isolated_points(point_clouds[i] * z_scale, 2 * neighb_dist_thresh)
        point_clouds[i] = point_clouds[i][isolated]
    z_scale = z_scale[2]
    # Add shifted reference point cloud
    if nb.has_page('register_initial_debug'):
        shift = nb.register_initial_debug.shift[t, r]
    else:
        shift = view_initial_shift(nb, t, r, return_shift=True)
    point_clouds = point_clouds + [point_clouds[1] + shift]

    # Add reference point cloud transformed by an affine transform
    transform_outlier = None
    if nb.has_page('register'):
        transform = nb.register.transform[t, r, c]
        if nb.has_page('register_debug'):
            # If particular tile/round/channel was found by regularised least squares
            transform_outlier = nb.register_debug.transform_outlier[t, r, c]
            if np.abs(transform_outlier).max() == 0:
                transform_outlier = None
    else:
        transform = get_single_affine_transform(config['register'], point_clouds[1], point_clouds[0], z_scale, z_scale,
                                                shift, neighb_dist_thresh, nb.basic_info.tile_centre)[0]

    if not nb.basic_info.is_3d:
        # use numpy not jax.numpy as reading in tiff is done in numpy.
        tile_sz = np.array([nb.basic_info.tile_sz, nb.basic_info.tile_sz, 1], dtype=np.int16)
    else:
        tile_sz = np.array([nb.basic_info.tile_sz, nb.basic_info.tile_sz, nb.basic_info.nz], dtype=np.int16)

    if transform_outlier is not None:
        point_clouds = point_clouds + [apply_transform(point_clouds[1], transform_outlier, nb.basic_info.tile_centre,
                                                       z_scale, tile_sz)[0]]

    point_clouds = point_clouds + [apply_transform(point_clouds[1], transform, nb.basic_info.tile_centre, z_scale,
                                                   tile_sz)[0]]
    pc_labels = [f'Imaging: r{r}, c{c}', f'Reference: r{r_ref}, c{c_ref}', f'Reference: r{r_ref}, c{c_ref} - Shift',
                 f'Reference: r{r_ref}, c{c_ref} - Affine']
    if transform_outlier is not None:
        pc_labels = pc_labels + [f'Reference: r{r_ref}, c{c_ref} - Regularized']
    view_point_clouds(point_clouds, pc_labels, neighb_dist_thresh, z_scale,
                      f'Transform of tile {t} to round {r}, channel {c}')
    plt.show()
