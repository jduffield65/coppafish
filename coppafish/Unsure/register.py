# from .. import register
import numpy as np
from typing import Tuple

from coppafish.Unsure import register as unsure_register
from coppafish import find_spots
from coppafish.setup.notebook import NotebookPage


def register(config: dict, nbp_basic: NotebookPage, spot_details: np.ndarray, spot_no: np.ndarray,
             start_transform: np.ndarray) -> Tuple[NotebookPage, NotebookPage]:
    """
    This finds the affine transforms to go from the ref round/channel to each imaging round/channel for every tile.
    It uses iterative closest point and the starting shifts found in `pipeline/register_initial.py`.

    See `'register'` and `'register_debug'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        config: Dictionary obtained from `'register'` section of config file.
        nbp_basic: `basic_info` notebook page
        spot_details: `int [n_spots x 3]`.
            `spot_details[s]` is `[ y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.spot_details`.
        spot_no: 'int[n_tiles x n_rounds x n_channels]'
            'spot_no[t,r,c]' is num_spots found on that [t,r,c]
        start_transform: n_tiles x n_rounds x n_channels x 4 x 3 array of initial starting affine fits.

    Returns:
        - `NotebookPage[register]` - Page contains the affine transforms to go from the ref round/channel to
            each imaging round/channel for every tile.
        - `NotebookPage[register_debug]` - Page contains information on how the affine transforms were calculated.
    """
    nbp = NotebookPage("register")
    nbp_debug = NotebookPage("register_debug")

    if nbp_basic.is_3d:
        neighb_dist_thresh = config['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['neighb_dist_thresh_2d']

    # scale spot yxz coordinates
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
    spot_yxz_ref = np.zeros(nbp_basic.n_tiles, dtype=object)
    spot_yxz_imaging = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels), dtype=object)
    n_matches_thresh = np.zeros_like(spot_yxz_imaging, dtype=float)

    for t in nbp_basic.use_tiles:
        spot_yxz_ref[t] = find_spots.spot_yxz(spot_details, t, nbp_basic.anchor_round, nbp_basic.anchor_channel, 
                                              spot_no) * np.array([1, 1, z_scale])
        for r in nbp_basic.use_rounds:
            for c in nbp_basic.use_channels:
                spot_yxz_imaging[t, r, c] = \
                    find_spots.spot_yxz(spot_details, t, r, c, spot_no) * np.array([1, 1, z_scale])
                if neighb_dist_thresh < 50:
                    # only keep isolated spots, those whose second neighbour is far away
                    isolated = find_spots.get_isolated_points(spot_yxz_imaging[t, r, c], 2 * neighb_dist_thresh)
                    spot_yxz_imaging[t, r, c] = spot_yxz_imaging[t, r, c][isolated, :]
                n_matches_thresh[t, r, c] = (config['matches_thresh_fract'] *
                                             np.min([spot_yxz_ref[t].shape[0], spot_yxz_imaging[t, r, c].shape[0]]))

    # get indices of tiles/rounds/channels used
    trc_ind = np.ix_(nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels)
    tr_ind = np.ix_(nbp_basic.use_tiles, nbp_basic.use_rounds)  # needed for av_shifts as no channel index

    n_matches_thresh[trc_ind] = np.clip(n_matches_thresh[trc_ind], config['matches_thresh_min'],
                                        config['matches_thresh_max'])
    n_matches_thresh = n_matches_thresh.astype(int)

    # Initialise variables. This includes spaces for tiles/rounds/channels not used
    final_transform = np.zeros_like(start_transform)
    n_matches = np.zeros_like(spot_yxz_imaging, dtype=int)
    error = np.zeros_like(spot_yxz_imaging, dtype=float)
    failed = np.zeros_like(spot_yxz_imaging, dtype=bool)
    converged = np.zeros_like(spot_yxz_imaging, dtype=bool)
    av_scaling = np.zeros((nbp_basic.n_channels, 3), dtype=float)
    av_shifts = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds, 3))
    transform_outliers = np.zeros_like(start_transform)

    # Deviation in scale/rotation is much less than permitted deviation in shift so boost scale reg constant.
    reg_constant_scale = np.sqrt(0.5 * config['regularize_constant'] * config['regularize_factor'])
    reg_constant_shift = np.sqrt(0.5 * config['regularize_constant'])

    # get ICP output only for tiles/rounds/channels that we are using
    final_transform[trc_ind], pcr_debug = \
        unsure_register.icp(spot_yxz_ref[nbp_basic.use_tiles], spot_yxz_imaging[trc_ind],
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

# nb = Notebook('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Initial scaling test/output/notebook.npz',
#               'C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Initial scaling test/E-2210-001_CP_settings_v2_z-scale.ini')
# register(nb.get_config()['register'], nb.basic_info, nb.find_spots.spot_details, nb.find_spots.spot_no,
#                                   nb.register_initial.shift, nb.register_initial.z_expansion_factor)