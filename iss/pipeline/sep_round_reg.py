from typing import List, Tuple
from .run import initialize_nb, run_extract, run_find_spots
from .. import setup, utils
from ..call_spots import get_non_duplicate
from ..stitch import compute_shift
from ..find_spots import get_isolated_points
from .. import pcr
from . import stitch
import numpy as np
import os
import warnings


def run_sep_round_reg(config_file: str, config_file_full: str, channels_to_save: List):
    """
    This runs the pipeline for a separate round up till the end of the stitching stage and then finds the
    affine transform that takes it to the anchor image of the full pipeline run.
    It then saves the corresponding transformed images for the channels of the separate round indicated by
    `channels_to_save`.

    Args:
        config_file: config_file: Path to config file for separate round.
            This should have only 1 round, that round being an anchor round and only one channel being used so
            filtering is only done on the anchor channel.
        config_file_full: Path to config file for full pipeline run, for which full notebook exists.
        channels_to_save: Channels of the separate round, that will be saved to the output directory in
            the same coordinate system as the anchor round of the full run.
    """
    # Get all information from full pipeline results - global spot positions and z scaling
    nb_full = initialize_nb(config_file_full)
    z_scale_full = nb_full.basic_info.pixel_size_z / nb_full.basic_info.pixel_size_xy
    global_yxz_full = nb_full.ref_spots.local_yxz + nb_full.stitch.tile_origin[nb_full.ref_spots.tile]

    # run pipeline to get as far as a set of global coordinates for the separate round anchor.
    nb = initialize_nb(config_file)
    config = setup.get_config(config_file)
    run_extract(nb, config)
    run_find_spots(nb, config)
    if not nb.has_page("stitch"):
        nbp_stitch = stitch(config['stitch'], nb.basic_info, nb.find_spots.spot_details)
        nb += nbp_stitch
    else:
        warnings.warn('stitch', utils.warnings.NotebookPageWarning)

    if not nb.has_page('reg_to_anchor_info'):
        # remove duplicate spots
        spot_local_yxz = nb.find_spots.spot_details[:, -3:]
        spot_tile = nb.find_spots.spot_details[:, 0]
        not_duplicate = get_non_duplicate(nb.stitch.tile_origin, nb.basic_info.use_tiles,
                                          nb.basic_info.tile_centrespot_local_yxz, spot_tile)
        global_yxz = spot_local_yxz[not_duplicate] + nb.stitch.tile_origin[spot_tile[not_duplicate]]

        # Only keep isolated points far from neighbour
        if nb.basic_info.is_3d:
            neighb_dist_thresh = config['register']['neighb_dist_thresh_3d']
        else:
            neighb_dist_thresh = config['register']['neighb_dist_thresh_2d']
        # scale z coordinate so in units of xy pixels as other 2 coordinates are.
        z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
        isolated = get_isolated_points(global_yxz * [1, 1, z_scale], 2 * neighb_dist_thresh)
        isolated_full = get_isolated_points(global_yxz_full * [1, 1, z_scale_full], 2 * neighb_dist_thresh)
        global_yxz = global_yxz[:, isolated]
        global_yxz_full = global_yxz_full[:, isolated_full]

        # Because applying transform to image, we don't do z-pixel conversion as would make final transform more
        # complicated. NOT SURE IF THIS IS BEST WAY!!
        z_scale = 1
        z_scale_full = 1

        # get initial shift from separate round to the full anchor image
        nbp = setup.NotebookPage('reg_to_anchor_info')
        nbp.shift, nbp.shift_score, nbp.shift_score_thresh = get_shift(nb.basic_info, config['register_initial'],
                                                                       global_yxz, global_yxz_full, z_scale, z_scale_full,
                                                                       nb.basic_info.is_3d)

        # Get affine transform from separate round to full anchor image
        nbp.transform, nbp.n_matches, nbp.error, nbp.is_converged = \
            get_affine_transform(config['register'], global_yxz, global_yxz_full, z_scale, z_scale_full,
                                 nb.shift, neighb_dist_thresh)
        nb += nbp  # save results of transform found
    else:
        nbp = nb.reg_to_anchor_info

    # save stitched images
    if nb.basic_info.is_3d:
        transform = nbp.transform
    else:
        transform = nbp.transform[[0,1,3], :-1]  # Need 2D 3 x 2 transform if 2D to save images.
    # save all the images
    for c in channels_to_save:
        im_file = os.path.join(nb.file_names.output_dir, f'sep_round_channel{c}_transformed.npz')
        if c == nb.basic_info.ref_channel:
            from_nd2 = False
        else:
            from_nd2 = True
        utils.npy.save_stitched(im_file, nb.file_names, nb.basic_info, nb.stitch.tile_origin, nb.basic_info.ref_round,
                                c, from_nd2, config['stitch']['save_image_zero_thresh'], transform)


def get_shift(config: dict, spot_yxz_base: np.ndarray, spot_yxz_transform: np.ndarray, z_scale_base: float,
              z_scale_transform: float, is_3d: bool) -> Tuple[np.ndarray, float, float]:
    """
    Find shift from base to transform.

    Args:
        config: register_initial section of config file corresponding to spot_yxz_base.
        spot_yxz_base: Point cloud want to find the shift from.
            spot_yxz_base[:, 2] is the z coordinate in units of z-pixels.
        spot_yxz_transform: Point cloud want to find the shift to.
            spot_yxz_transform[:, 2] is the z coordinate in units of z-pixels.
        z_scale_base: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        z_scale_transform: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        is_3d: Whether pipeline is 3D or not.

    Returns:
        - `shift` - `float [shift_y, shift_x, shift_z]`.
            Best shift found.
        - `shift_score` - `float`.
            Score of best shift found.
        - `min_score` - `float`.
            Threshold score that was calculated, i.e. range of shifts searched changed until score exceeded this.
    """

    coords = ['y', 'x', 'z']
    shifts = {}
    for i in range(len(coords)):
        shifts[coords[i]] = np.arange(config['shift_min'][i],
                                      config['shift_max'][i] +
                                      config['shift_step'][i] / 2, config['shift_step'][i]).astype(int)
    if not is_3d:
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0
        shifts['z'] = np.array([0], dtype=int)
    shift, shift_score, shift_score_thresh = \
        compute_shift(spot_yxz_base, spot_yxz_transform,
                      config['shift_score_thresh'], config['shift_score_thresh_multiplier'],
                      config['shift_score_thresh_min_dist'], config['shift_score_thresh_max_dist'],
                      config['neighb_dist_thresh'], shifts['y'], shifts['x'], shifts['z'],
                      config['shift_widen'], config['shift_max_range'], [z_scale_base, z_scale_transform],
                      config['nz_collapse'], config['shift_step'][2])
    return shift, shift_score, shift_score_thresh


def get_affine_transform(config: dict, spot_yxz_base: np.ndarray, spot_yxz_transform: np.ndarray, z_scale_base: float,
                         z_scale_transform: float, initial_shift: np.ndarray,
                         neighb_dist_thresh: float) -> Tuple[np.ndarray, int, float, bool]:
    """
    Finds the affine transform taking spot_yxz_base to spot_yxz_transform.

    Args:
        config: register section of config file corresponding to spot_yxz_base.
        spot_yxz_base: Point cloud want to find the shift from.
            spot_yxz_base[:, 2] is the z coordinate in units of z-pixels.
        spot_yxz_transform: Point cloud want to find the shift to.
            spot_yxz_transform[:, 2] is the z coordinate in units of z-pixels.
        z_scale_base: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        z_scale_transform: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        initial_shift: Shift to be used as starting point to find affine transfom.
        neighb_dist_thresh: Distance between 2 points must be less than this to be constituted a match.

    Returns:
        - `transform` - `float [4 x 3]`.
            `transform` is the final affine transform found.
        - `n_matches` - Number of matches found for each transform.
        - `error` - Average distance between neighbours below `neighb_dist_thresh`.
        - `is_converged` - `False` if max iterations reached before transform converged.
    """
    n_tiles = 1
    n_channels = 1
    n_rounds = 1
    initial_shift = np.asarray(initial_shift).reshape(n_tiles, n_rounds, -1)
    n_matches_thresh = config['matches_thresh_fract'] * np.min(
        [spot_yxz_base.shape[0], spot_yxz_transform.shape[0]])
    n_matches_thresh = np.clip(n_matches_thresh, config['matches_thresh_min'], config['matches_thresh_max'])
    n_matches_thresh = n_matches_thresh.astype(int)
    initial_shift = initial_shift * [1, 1, z_scale_base]
    start_transform = pcr.transform_from_scale_shift(np.ones((n_channels, 3)), initial_shift)
    final_transform, pcr_debug = \
        pcr.iterate(spot_yxz_base * [1, 1, z_scale_base], spot_yxz_transform * [1, 1, z_scale_transform],
                    start_transform, config['n_iter'], neighb_dist_thresh,
                    n_matches_thresh, config['scale_dev_thresh'], config['shift_dev_thresh'],
                    None, None)
    return final_transform.squeeze(), int(pcr_debug['n_matches']), float(pcr_debug['error']
                                                                         ), bool(pcr_debug['is_converged'])
