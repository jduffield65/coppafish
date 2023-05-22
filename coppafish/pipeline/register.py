import os
import pickle
import numpy as np
from tqdm import tqdm
from ..setup import NotebookPage
from ..find_spots import spot_yxz, spot_isolated
from ..register.base import icp, regularise_transforms, round_registration, channel_registration
from ..register.preprocessing import compose_affine, zyx_to_yxz_affine, load_reg_data


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

    # Break algorithm up into initialisation and then 3 parts.

    # Part 0: Initialise variables and load in data from previous runs of the software
    # Part 1: Generate subvolumes, use these in a regression to obtain an initial estimate for affine transform
    # Part 2: Compare the transforms across tiles to remove outlier transforms
    # Part 3: Correct the initial transform guesses with an ICP algorithm

    # Part 0: Initialisation
    # Initialise frequently used variables
    nbp, nbp_debug = NotebookPage("register"), NotebookPage("register_debug")
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels

    # Initialise variables for ICP step
    if nbp_basic.is_3d:
        neighb_dist_thresh = config['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['neighb_dist_thresh_2d']

    # Load in registration data from previous runs of the software
    registration_data = load_reg_data(nbp_file, nbp_basic, config)
    uncompleted_tiles = np.setdiff1d(use_tiles, registration_data['round_registration']['tiles_completed'])

    # Part 1: Initial affine transform
    # Start with round registration
    with tqdm(total=len(uncompleted_tiles)) as pbar:
        pbar.set_description(f"Running initial round registration on all tiles")
        for t in uncompleted_tiles:
            registration_data = round_registration(nbp_file, nbp_basic, config, registration_data, t, pbar)

    # Now do channel registration
    uncompleted_tiles = np.setdiff1d(use_tiles, registration_data['channel_registration']['tiles_completed'])
    with tqdm(total=len(uncompleted_tiles)) as pbar:
        pbar.set_description(f"Running initial channel registration on all tiles")
        for t in uncompleted_tiles:
            registration_data = channel_registration(nbp_file, nbp_basic, config, registration_data, t, pbar)

    # Part 2: Regularisation
    registration_data = regularise_transforms(registration_data=registration_data,
                                              tile_origin=np.roll(tile_origin, 1, axis=1),
                                              residual_threshold=config['residual_thresh'],
                                              use_tiles=nbp_basic.use_tiles,
                                              use_rounds=nbp_basic.use_rounds,
                                              use_channels=nbp_basic.use_channels)

    # Now combine all of these into single subvol transform array via composition
    for t in use_tiles:
        for r in use_rounds:
            for c in use_channels:
                registration_data['initial_transform'][t, r, c] = \
                    zyx_to_yxz_affine(compose_affine(registration_data['channel_registration']['channel_transform'][t, c],
                                                     registration_data['round_registration']['round_transform'][t, r]))
    # Now save registration data externally
    with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'wb') as f:
        pickle.dump(registration_data, f)

    # Part 3: ICP
    icp_transform = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    n_matches = np.zeros((n_tiles, n_rounds, n_channels, config['n_iter']))
    mse = np.zeros((n_tiles, n_rounds, n_channels, config['n_iter']))
    converged = np.zeros((n_tiles, n_rounds, n_channels), dtype=bool)

    # Create a progress bar for the ICP step
    with tqdm(total=len(use_tiles) * len(use_rounds) * len(use_channels)) as pbar:
        pbar.set_description(f"Running ICP on all tiles")
        for t in use_tiles:
            # Do not do ICP if done in previous run
            if 'icp' in registration_data.keys():
                break
            ref_spots_t = spot_yxz(nbp_find_spots.spot_details, t, nbp_basic.anchor_round, nbp_basic.anchor_channel,
                                   nbp_find_spots.spot_no)
            for r in use_rounds:
                for c in use_channels:
                    pbar.set_postfix({"Tile": t, "Round": r, "Channel": c})
                    # Only do ICP on non-degenerate cells with more than 100 spots
                    if nbp_find_spots.spot_no[t, r, c] > 100:
                        imaging_spots_trc = spot_yxz(nbp_find_spots.spot_details, t, r, c, nbp_find_spots.spot_no)
                        icp_transform[t, r, c], n_matches[t, r, c], mse[t, r, c], converged[t, r, c] = icp(
                            yxz_base=ref_spots_t,
                            yxz_target=imaging_spots_trc,
                            dist_thresh=neighb_dist_thresh,
                            start_transform=registration_data['initial_transform'][t, r, c],
                            n_iters=50,
                            robust=False)
                    else:
                        # Otherwise just use the starting transform
                        icp_transform[t, r, c] = registration_data['initial_transform'][t, r, c]
                    pbar.update(1)
    # Save ICP data
    registration_data['icp'] = {'icp_transform': icp_transform, 'n_matches': n_matches, 'mse': mse,
                                'converged': converged}
    # Save registration data externally
    with open(os.path.join(nbp_file.output_dir, 'registration_data.pkl'), 'wb') as f:
        pickle.dump(registration_data, f)

    # Add round statistics to debugging page.
    # First add the round registration statistics
    nbp_debug.position = registration_data['round_registration']['position']
    nbp_debug.round_shift = registration_data['round_registration']['round_shift']
    nbp_debug.round_shift_corr = registration_data['round_registration']['round_shift_corr']
    nbp_debug.round_transform_raw = registration_data['round_registration']['round_transform_raw']

    # Now add the channel registration statistics
    nbp_debug.channel_shift = registration_data['channel_registration']['channel_shift']
    nbp_debug.reference_round = registration_data['channel_registration']['reference_round']
    nbp_debug.channel_shift_corr = registration_data['channel_registration']['channel_shift_corr']
    nbp_debug.channel_transform_raw = registration_data['channel_registration']['channel_transform_raw']

    # Now add the ICP statistics
    nbp_debug.mse = registration_data['icp']['mse']
    nbp_debug.n_matches = registration_data['icp']['n_matches']
    nbp_debug.converged = registration_data['icp']['converged']

    # Now add relevant information to the nbp object
    nbp.round_transform = registration_data['round_registration']['round_transform']
    nbp.channel_transform = registration_data['channel_registration']['channel_transform']
    nbp.initial_transform = registration_data['initial_transform']
    nbp.transform = registration_data['icp']['icp_transform']

    return nbp, nbp_debug
