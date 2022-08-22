# from .. import register
from ..register import icp, transform_from_scale_shift
from ..find_spots import spot_yxz, get_isolated_points
import numpy as np
from ..setup.notebook import NotebookPage
from typing import Tuple


def register(config: dict, nbp_basic: NotebookPage, spot_details: np.ndarray,
             initial_shift: np.ndarray) -> Tuple[NotebookPage, NotebookPage]:
    """
    This finds the affine transforms to go from the ref round/channel to each imaging round/channel for every tile.
    It uses iterative closest point and the starting shifts found in `pipeline/register_initial.py`.

    See `'register'` and `'register_debug'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        config: Dictionary obtained from `'register'` section of config file.
        nbp_basic: `basic_info` notebook page
        spot_details: `int [n_spots x 7]`.
            `spot_details[s]` is `[tile, round, channel, isolated, y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.spot_details`.
        initial_shift: `int [n_tiles x n_rounds x 3]`.
            `initial_shift[t, r]` is the yxz shift found that is applied to tile `t`, `ref_round` to take it to
            tile `t`, round `r`. Units: `[yx_pixels, yx_pixels, z_pixels]`.
            This is saved in the `register_initial_debug` notebook page i.e. `nb.register_initial_debug.shift`.

    Returns:
        - `NotebookPage[register]` - Page contains the affine transforms to go from the ref round/channel to
            each imaging round/channel for every tile.
        - `NotebookPage[register_debug]` - Page contains information on how the affine transforms were calculated.
    """
    nbp = NotebookPage("register")
    nbp_debug = NotebookPage("register_debug")
    nbp.initial_shift = initial_shift.copy()

    if nbp_basic.is_3d:
        neighb_dist_thresh = config['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['neighb_dist_thresh_2d']

    # centre and scale spot yxz coordinates
    z_scale = [1, 1, nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy]
    spot_yxz_ref = np.zeros(nbp_basic.n_tiles, dtype=object)
    spot_yxz_imaging = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels), dtype=object)
    n_matches_thresh = np.zeros_like(spot_yxz_imaging, dtype=float)
    initial_shift = initial_shift.astype(float)
    for t in nbp_basic.use_tiles:
        spot_yxz_ref[t] = spot_yxz(spot_details, t, nbp_basic.ref_round, nbp_basic.ref_channel)
        spot_yxz_ref[t] = (spot_yxz_ref[t] - nbp_basic.tile_centre) * z_scale
        for r in nbp_basic.use_rounds:
            initial_shift[t, r] = initial_shift[t, r] * z_scale  # put z initial shift into xy pixel units
            for c in nbp_basic.use_channels:
                spot_yxz_imaging[t, r, c] = spot_yxz(spot_details, t, r, c)
                spot_yxz_imaging[t, r, c] = (spot_yxz_imaging[t, r, c] - nbp_basic.tile_centre) * z_scale
                if neighb_dist_thresh < 50:
                    # only keep isolated spots, those whose second neighbour is far away
                    isolated = get_isolated_points(spot_yxz_imaging[t, r, c], 2 * neighb_dist_thresh)
                    spot_yxz_imaging[t, r, c] = spot_yxz_imaging[t, r, c][isolated, :]
                n_matches_thresh[t, r, c] = (config['matches_thresh_fract'] *
                                             np.min([spot_yxz_ref[t].shape[0], spot_yxz_imaging[t, r, c].shape[0]]))

    # get indices of tiles/rounds/channels used
    trc_ind = np.ix_(nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels)
    tr_ind = np.ix_(nbp_basic.use_tiles, nbp_basic.use_rounds)  # needed for av_shifts as no channel index

    n_matches_thresh[trc_ind] = np.clip(n_matches_thresh[trc_ind], config['matches_thresh_min'],
                                        config['matches_thresh_max'])
    n_matches_thresh = n_matches_thresh.astype(int)

    # Initialise variables obtain from PCR algorithm. This includes spaces for tiles/rounds/channels not used
    start_transform = transform_from_scale_shift(np.ones((nbp_basic.n_channels, 3)), initial_shift)
    final_transform = np.zeros_like(start_transform)
    n_matches = np.zeros_like(spot_yxz_imaging, dtype=int)
    error = np.zeros_like(spot_yxz_imaging, dtype=float)
    failed = np.zeros_like(spot_yxz_imaging, dtype=bool)
    converged = np.zeros_like(spot_yxz_imaging, dtype=bool)
    av_scaling = np.zeros((nbp_basic.n_channels, 3), dtype=float)
    av_shifts = np.zeros_like(initial_shift)
    transform_outliers = np.zeros_like(start_transform)

    # Deviation in scale/rotation is much less than permitted deviation in shift so boost scale reg constant.
    reg_constant_scale = np.sqrt(0.5 * config['regularize_constant'] * config['regularize_factor'])
    reg_constant_shift = np.sqrt(0.5 * config['regularize_constant'])

    # get ICP output only for tiles/rounds/channels that we are using
    final_transform[trc_ind], pcr_debug = \
        icp(spot_yxz_ref[nbp_basic.use_tiles], spot_yxz_imaging[trc_ind],
            start_transform[trc_ind], config['n_iter'], neighb_dist_thresh,
            n_matches_thresh[trc_ind], config['scale_dev_thresh'], config['shift_dev_thresh'],
            reg_constant_scale, reg_constant_shift)

    # save debug info at correct tile, round, channel index
    n_matches[trc_ind] = pcr_debug['n_matches']
    error[trc_ind] = pcr_debug['error']
    failed[trc_ind] = pcr_debug['failed']
    converged[trc_ind] = pcr_debug['is_converged']
    av_scaling[nbp_basic.use_channels] = pcr_debug['av_scaling']
    av_shifts[tr_ind] = pcr_debug['av_shifts']
    transform_outliers[trc_ind] = pcr_debug['transforms_outlier']

    # add to notebook
    nbp.transform = final_transform
    nbp_debug.n_matches = n_matches
    nbp_debug.n_matches_thresh = n_matches_thresh
    nbp_debug.error = error
    nbp_debug.failed = failed
    nbp_debug.converged = converged
    nbp_debug.av_scaling = av_scaling
    nbp_debug.av_shifts = av_shifts
    nbp_debug.transform_outlier = transform_outliers

    return nbp, nbp_debug
