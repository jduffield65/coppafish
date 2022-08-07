import numpy as np
from .shift import view_register_search
from ...find_spots import spot_yxz, get_isolated_points
from ...register import get_single_affine_transform
from ...spot_colors.base import apply_transform
from ..stitch import view_point_clouds
from ...setup import Notebook
from typing import Optional, List
import matplotlib.pyplot as plt


def view_icp(nb: Notebook, t: int, r: int, c: int):
    """
    Function to plot results of iterative closest point to find affine transform between
    `ref_round/ref_channel` and round `r`, channel `c` for tile `t`.
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
    # only keep isolated spots, those whose second neighbour is far away
    # Do this only for imaging point cloud as that is what is done in pipeline/register
    isolated = get_isolated_points(point_clouds[0] * z_scale, 2 * neighb_dist_thresh)
    point_clouds[0] = point_clouds[0][isolated]

    # 2nd is untransformed reference point cloud
    r_ref = nb.basic_info.ref_round
    c_ref = nb.basic_info.ref_channel
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref)]
    z_scale = z_scale[2]

    # Add shifted reference point cloud
    if nb.has_page('register_initial'):
        shift = nb.register_initial.shift[t, r]
    else:
        shift = view_register_search(nb, t, r, return_shift=True)
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
        start_transform = np.eye(4, 3)  # no scaling just shift to start off icp
        start_transform[3] = shift * [1, 1, z_scale]
        transform = get_single_affine_transform(point_clouds[1], point_clouds[0], z_scale, z_scale,
                                                start_transform, neighb_dist_thresh, nb.basic_info.tile_centre)[0]

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


def view_icp_reg(nb: Notebook, t: int, r: int, c: int, reg_constant_rot: List, reg_constant_shift: List,
                 reg_transform: Optional[np.ndarray] = None, start_transform: Optional[np.ndarray] = None):
    config = nb.get_config()
    if nb.basic_info.is_3d:
        neighb_dist_thresh = config['register']['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['register']['neighb_dist_thresh_2d']
    z_scale = [1, 1, nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy]
    point_clouds = []
    # 1st point cloud is imaging one as does not change
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r, c)]
    # only keep isolated spots, those whose second neighbour is far away
    # Do this only for imaging point cloud as that is what is done in pipeline/register
    isolated = get_isolated_points(point_clouds[0] * z_scale, 2 * neighb_dist_thresh)
    point_clouds[0] = point_clouds[0][isolated]

    # 2nd is untransformed reference point cloud
    r_ref = nb.basic_info.ref_round
    c_ref = nb.basic_info.ref_channel
    point_clouds = point_clouds + [spot_yxz(nb.find_spots.spot_details, t, r_ref, c_ref)]
    z_scale = z_scale[2]



