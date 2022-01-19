from .. import utils, setup
from .. import pcr
from ..find_spots import spot_yxz
import numpy as np
from sklearn.neighbors import NearestNeighbors


def run_register(config, nbp_basic, spot_details, initial_shift):
    nbp = setup.NotebookPage("register")
    nbp_debug = setup.NotebookPage("register_debug")
    nbp['initial_shift'] = initial_shift.copy()

    # centre and scale spot yxz coordinates
    z_scale = [1, 1, nbp_basic['pixel_size_z'] / nbp_basic['pixel_size_xy']]
    spot_yxz_ref = np.zeros(nbp_basic['n_tiles'], dtype=object)
    spot_yxz_imaging = np.zeros((nbp_basic['n_tiles'], nbp_basic['n_rounds'], nbp_basic['n_channels']), dtype=object)
    n_matches_thresh = np.zeros_like(spot_yxz_imaging, dtype=float)
    initial_shift = initial_shift.astype(float)
    for t in nbp_basic['use_tiles']:
        spot_yxz_ref[t] = spot_yxz(spot_details, t, nbp_basic['ref_round'], nbp_basic['ref_channel'])
        spot_yxz_ref[t] = (spot_yxz_ref[t] - nbp_basic['tile_centre']) * z_scale
        for r in nbp_basic['use_rounds']:
            initial_shift[t, r] = initial_shift[t, r] * z_scale  # put z initial shift into xy pixel units
            for c in nbp_basic['use_channels']:
                spot_yxz_imaging[t, r, c] = spot_yxz(spot_details, t, r, c)
                spot_yxz_imaging[t, r, c] = (spot_yxz_imaging[t, r, c] - nbp_basic['tile_centre']) * z_scale
                if config['neighb_dist_thresh'] < 50:
                    # only keep isolated spots, those whose second neighbour is far away
                    tree = NearestNeighbors(n_neighbors=2).fit(spot_yxz_imaging[t, r, c])
                    distances, _ = tree.kneighbors(spot_yxz_imaging[t, r, c])
                    isolated = distances[:, 1] > 2 * config['neighb_dist_thresh']
                    spot_yxz_imaging[t, r, c] = spot_yxz_imaging[t, r, c][isolated, :]
                n_matches_thresh[t, r, c] = (config['matches_thresh_fract'] *
                                             np.min([spot_yxz_ref[t].shape[0], spot_yxz_imaging[t, r, c].shape[0]]))

    # get indices of tiles/rounds/channels used
    t_ind, r_ind, c_ind = utils.multi_array_ind(nbp_basic['use_tiles'], nbp_basic['use_rounds'],
                                                nbp_basic['use_channels'])
    n_matches_thresh[t_ind, r_ind, c_ind] = np.clip(n_matches_thresh[t_ind, r_ind, c_ind], config['matches_thresh_min'],
                                                    config['matches_thresh_max'])
    n_matches_thresh = n_matches_thresh.astype(int)

    # Initialise variables obtain from PCR algorithm. This includes spaces for tiles/rounds/channels not used
    start_transform = pcr.transform_from_scale_shift(np.ones((nbp_basic['n_channels'], 3)), initial_shift)
    final_transform = np.zeros_like(start_transform)
    n_matches = np.zeros_like(spot_yxz_imaging, dtype=int)
    error = np.zeros_like(spot_yxz_imaging, dtype=float)
    failed = np.zeros_like(spot_yxz_imaging, dtype=bool)
    converged = np.zeros_like(spot_yxz_imaging, dtype=bool)
    av_scaling = np.zeros((nbp_basic['n_channels'], 3), dtype=float)
    av_shifts = np.zeros_like(initial_shift)
    transform_outliers = np.zeros_like(start_transform)

    # get PCR output only for tiles/rounds/channels that we are using
    final_transform[:, :, t_ind, r_ind, c_ind], n_matches[t_ind, r_ind, c_ind], error[t_ind, r_ind, c_ind], \
    failed[t_ind, r_ind, c_ind], converged[t_ind, r_ind, c_ind], av_scaling[nbp_basic['use_channels']], \
    av_shifts[t_ind[:, :, 0], r_ind[:, :, 0]], transform_outliers[:, :, t_ind, r_ind, c_ind] = \
        pcr.iterate(spot_yxz_ref[nbp_basic['use_tiles']], spot_yxz_imaging[t_ind, r_ind, c_ind],
                    start_transform[:, :, t_ind, r_ind, c_ind], config['n_iter'], config['neighb_dist_thresh'],
                    n_matches_thresh[t_ind, r_ind, c_ind], config['scale_dev_thresh'], config['shift_dev_thresh'],
                    config['regularize_constant_scale'], config['regularize_constant_shift'])

    # add to notebook
    nbp['transform'] = final_transform
    nbp_debug['n_matches'] = n_matches
    nbp_debug['n_matches_thresh'] = n_matches_thresh
    nbp_debug['error'] = error
    nbp_debug['failed'] = failed
    nbp_debug['converged'] = converged
    nbp_debug['av_scaling'] = av_scaling
    nbp_debug['av_shifts'] = av_shifts
    nbp_debug['transform_outlier'] = transform_outliers

    return nbp, nbp_debug
