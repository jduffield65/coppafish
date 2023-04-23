import os
import pickle
import numpy as np
from tqdm import tqdm
from ..setup import NotebookPage
from ..find_spots import spot_yxz
from ..register.base import icp, regularise_transforms, subvolume_registration
from ..register.preprocessing import compose_affine, reformat_affine, load_reg_data


def register(nbp_basic: NotebookPage, nbp_file: NotebookPage, nbp_find_spots: NotebookPage, config: dict,
             tile_origin: np.ndarray):
    """
    Registration pipeline. Returns register Notebook Page.
    Finds affine transforms by using linear regression to find the best matrix (in the least squares sense) taking a
    bunch of points in one image to corresponding points in another. These shifts are found with a phase cross
    correlation algorithm.

    To get greater precision with this algorithm, we update these transforms with an iterative closest point algorithm.

    Args:
        nbp_basic: (NotebookPage) Basic Info notebook page
        nbp_file: (NotebookPage) File Names notebook page
        nbp_find_spots: (NotebookPage) Find Spots notebook page
        config: Register part of the config dictionary
        tile_origin: n_tiles x 3 ndarray of tile origins

    Returns:
        nbp: (NotebookPage) Register notebook page
        nbp_debug: (NotebookPage) Register_debug notebook page
    """

    # Break algorithm up into 1 + 2 parts.

    # Part 0: Initialise variables and load in data from previous runs of the software
    # Part 1: Generate the positions and their associated shifts for the round and channel registration. Save these in
    # the external dictionary registration_data.pkl so that we can keep the results if the algorithm crashes at any
    # point before reaching the end.
    # Part 2: correct these guesses with an ICP algorithm

    # Part 0: Initialisation
    # Initialise frequently used variables
    nbp, nbp_debug = NotebookPage("register"), NotebookPage("register_debug")
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy

    # Initialise variables for ICP step
    if nbp_basic.is_3d:
        neighb_dist_thresh = config['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['neighb_dist_thresh_2d']

    # Collect any data that we have already run
    registration_data = load_reg_data(nbp_file, nbp_basic, config)
    uncompleted_tiles = [t for t in use_tiles if t not in registration_data['tiles_completed']]

    # Part 1: Compute subvolume shifts
    with tqdm(total=len(uncompleted_tiles)) as pbar:
        pbar.set_description(f"Computing shifts for all subvolumes")
        for t in uncompleted_tiles:
            registration_data = subvolume_registration(nbp_file, nbp_basic, config, registration_data, t, pbar)
            pbar.update(1)

    # Now that we have all transformations, regularise and combine into a transform for each t, r, c
    registration_data['round_transform'] = regularise_transforms(transform=registration_data['round_transform'],
                                                                 residual_threshold=config['residual_threshold'],
                                                                 tile_origin=tile_origin.copy(),
                                                                 use_tiles=nbp_basic.use_tiles,
                                                                 rc_use=nbp_basic.use_rounds)
    registration_data['channel_transform'] = regularise_transforms(transform=registration_data['channel_transform'],
                                                                   residual_threshold=config['residual_threshold'],
                                                                   tile_origin=tile_origin.copy(),
                                                                   use_tiles=nbp_basic.use_tiles,
                                                                   rc_use=nbp_basic.use_rounds)
    # Initialise new key in registration data for subvol_transform
    registration_data['subvol_transform'] = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    # Begin combination of these into single subvol transform array
    for t in use_tiles:
        for r in use_rounds:
            for c in use_channels:
                registration_data['subvol_transform'][t, r, c] = \
                    reformat_affine(compose_affine(registration_data['channel_transform'][t, c],
                                                   registration_data['round_transform'][t, r]), z_scale)
    # Now save registration data externally
    with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'wb') as f:
        pickle.dump(registration_data, f)

    # Part 2: ICP
    # This algorithm works well, but we need pixel level registration. For this reason, correct this shift using ICP
    icp_transform = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    n_matches = np.zeros((n_tiles, n_rounds, n_channels, config['n_iter']))
    mse = np.zeros((n_tiles, n_rounds, n_channels, config['n_iter']))
    for t in use_tiles:
        ref_spots_t = spot_yxz(nbp_find_spots.spot_details, t, nbp_basic.ref_round, nbp_basic.ref_channel,
                               nbp_find_spots.spot_no)
        for r in use_rounds:
            for c in use_channels:
                # Only do ICP on non-degenerate cells with more than 100 spots
                if nbp_find_spots.spot_no[t, r, c] > 100:
                    imaging_spots_trc = spot_yxz(nbp_find_spots.spot_details, t, r, c, nbp_find_spots.spot_no)
                    icp_transform[t, r, c], n_matches[t, r, c], mse[t, r, c] = icp(yxz_base=ref_spots_t,
                                                                                   yxz_target=imaging_spots_trc,
                                                                                   dist_thresh=neighb_dist_thresh,
                                                                                   start_transform=registration_data
                                                                                   ['subvol_transform'][t, r, c],
                                                                                   n_iters=50, robust=False)
                else:
                    # Otherwise just use the starting transform
                    icp_transform[t, r, c] = registration_data['subvol_transform'][t, r, c]

    # Add convergence statistics to the debug notebook page.
    nbp_debug.n_matches = n_matches
    nbp_debug.mse = mse
    # These are not in use now but may be useful in the future
    nbp_debug.converged = None
    nbp_debug.failed = None
    nbp_debug.av_scaling = None
    nbp_debug.av_shifts = None
    nbp_debug.transform_outlier = None

    # add to register page of notebook
    nbp.subvol_transform = registration_data['subvol_transform']
    nbp.transform = icp_transform
    # Add round registration data
    nbp.round_shift = registration_data['round_shift']
    nbp.round_shift_corr = registration_data['round_shift_corr']
    nbp.round_transform = registration_data['round_transform']
    # Add channel registration data
    nbp.channel_shift = registration_data['channel_shift']
    nbp.channel_shift_corr = registration_data['channel_shift_corr']
    nbp.channel_transform = registration_data['channel_transform']

    return nbp, nbp_debug
