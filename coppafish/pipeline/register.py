import os
import numpy as np
from tqdm import tqdm
from skimage.filters import sobel
from skimage.exposure import match_histograms
from ..setup import NotebookPage
from ..utils.npy import load_tile
from ..find_spots import spot_yxz, get_isolated_points
from ..register.base import find_shift_array, ols_regression_robust, icp, regularise_transforms
from ..register.preprocessing import split_3d_image, compose_affine, invert_affine, reformat_affine


def register(nbp_basic: NotebookPage, nbp_file: NotebookPage, config: dict, spot_details: np.ndarray,
             spot_no: np.ndarray, tile_origin):

    """
    Registration pipeline. Returns register Notebook Page.
    Finds affine transforms by using linear regression to find the best matrix (in the least squares sense) taking a
    bunch of points in one image to corresponding points in another. These shifts are found with a phase cross
    correlation algorithm.

    To get greater precision with this algorithm, we update these transforms with an iterative closest point algorithm.

    Args:
        nbp_basic: (NotebookPage) Basic Info notebook page
        nbp_file: (NotebookPage) File Names notebook page
        config: Register part of the config dictionary
        spot_details: `int [n_spots x 3]`.
            `spot_details[s]` is `[ y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.spot_details`.
        spot_no: 'int[n_tiles x n_rounds x n_channels]'
            'spot_no[t,r,c]' is num_spots found on that [t,r,c]

    Returns:
        nbp: (NotebookPage) Register notebook page
        nbp_debug: (NotebookPage) Register_debug notebook page
    """

    # Break algorithm up into 3 parts.

    # Part 1: Generate the positions and their associated shifts for the round and channel registration. Save these in
    # the notebook.
    # Part 2: Compute the affine transform from this data, using our regression method of choice
    # Part 3: correct these guesses with an ICP algorithm

    # Part 0: Initialisation
    # Initialise frequently used variables
    nbp = NotebookPage("register")
    nbp_debug = NotebookPage("register_debug")
    config = config["register"]
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_tiles, n_rounds, n_channels = nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels
    r_ref, c_ref = nbp_basic.anchor_round, nbp_basic.anchor_channel
    n_iters = config['n_iter']
    n_matches, error = np.zeros((n_tiles, n_rounds, n_channels, n_iters)), \
                       np.zeros((n_tiles, n_rounds, n_channels, n_iters))
    # Next it's important to include z-scale as our affine transform will need to have the z-shift saved in units of xy
    # pixels
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
    # Also note the round we use for the channel shifts
    r_mid = n_rounds//2
    # Now initialise the registration parameters specified in the config file
    z_subvols, y_subvols, x_subvols = config['z_subvols'], config['y_subvols'], config['x_subvols']
    z_box, y_box, x_box = config['z_box'], config['y_box'], config['x_box']
    spread = np.array(config['spread'])

    # Initialise variables for ICP step
    if nbp_basic.is_3d:
        neighb_dist_thresh = config['neighb_dist_thresh_3d']
    else:
        neighb_dist_thresh = config['neighb_dist_thresh_2d']
    # Load and scale spot yxz coordinates
    spot_yxz_ref = np.zeros(n_tiles, dtype=object)
    spot_yxz_imaging = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)

    # Finally, initialise all data to be saved to the notebook page.
    # If this page has crashed previously then we will just use the shifts that have been computed
    # round_transforms is the affine transform from (r_ref, c_ref) to (r, c_ref) for all r in use
    round_transform = np.zeros((n_tiles, n_rounds, 3, 4))
    # channel_transform is the affine transforms from (r_mid, c_ref) to (r_mid c) for all c in use
    channel_transform = np.zeros((n_tiles, n_channels, 3, 4))
    # These will then be combined into a single array by affine composition, reformatted for compatibility later in code
    transform = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    # Not sure if we will keep this
    final_transform = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    # The next 4 arrays store the shifts and positions of the subvolumes between the images we are registering
    round_shift = np.zeros((n_tiles, n_rounds, z_subvols, y_subvols, x_subvols, 3))
    round_position = np.zeros((n_tiles, n_rounds, z_subvols, y_subvols, x_subvols, 3))
    channel_shift = np.zeros((n_tiles, n_channels, z_subvols, y_subvols, x_subvols, 3))
    channel_position = np.zeros((n_tiles, n_channels, z_subvols, y_subvols, x_subvols, 3))
    # Load in files if we have them
    if os.path.isfile(os.path.join(nbp_file.output_dir, 'round_shift.npy')):
        round_shift = np.load(os.path.join(nbp_file.output_dir, 'round_shift.npy'))
    if os.path.isfile(os.path.join(nbp_file.output_dir, 'round_position.npy')):
        round_position = np.load(os.path.join(nbp_file.output_dir, 'round_position.npy'))
    if os.path.isfile(os.path.join(nbp_file.output_dir, 'channel_shift.npy')):
        channel_shift = np.load(os.path.join(nbp_file.output_dir, 'channel_shift.npy'))
    if os.path.isfile(os.path.join(nbp_file.output_dir, 'channel_position.npy')):
        channel_position = np.load(os.path.join(nbp_file.output_dir, 'channel_position.npy'))

    # Get tiles that channel_shift has already been run on
    tiles_completed = list(set(np.argwhere(channel_shift)[:, 0]))
    uncompleted_tiles = [t for t in use_tiles if t not in tiles_completed]

    # Part 1: Compute subvolume shifts

    # For each tile, only need to do this for the n_rounds + channels not equal to ref channel
    with tqdm(total=len(uncompleted_tiles)) as pbar:
        pbar.set_description(f"Computing shifts for all subvolumes")
        for t in uncompleted_tiles:
            # Load in the anchor npy volume, only need to do this once per tile
            anchor_image_unfiltered = load_tile(nbp_file, nbp_basic, t, r_ref, c_ref)
            # Save the unfiltered version as well for histogram matching later
            anchor_image = sobel(anchor_image_unfiltered)

            # Software was written for z y x, so change it from y x z
            anchor_image = np.swapaxes(anchor_image, 0, 2)
            anchor_image = np.swapaxes(anchor_image, 1, 2)
            anchor_image_unfiltered = np.swapaxes(anchor_image_unfiltered, 0, 2)
            anchor_image_unfiltered = np.swapaxes(anchor_image_unfiltered, 1, 2)

            for r in use_rounds:
                pbar.set_postfix({'tile': f'{t}', 'round': f'{r}'})

                # Load in imaging npy volume.
                target_image = sobel(load_tile(nbp_file, nbp_basic, t, r, c_ref))
                target_image = np.swapaxes(target_image, 0, 2)
                target_image = np.swapaxes(target_image, 1, 2)

                # next we split image into overlapping cuboids
                subvol_base, position = split_3d_image(image=anchor_image, z_subvolumes=z_subvols,
                                                       y_subvolumes=y_subvols, x_subvolumes=x_subvols,
                                                       z_box=z_box, y_box=y_box, x_box=x_box)
                subvol_target, _ = split_3d_image(image=target_image, z_subvolumes=z_subvols, y_subvolumes=y_subvols,
                                                  x_subvolumes=x_subvols, z_box=z_box, y_box=y_box, x_box=x_box)

                # Find the subvolume shifts
                shift = find_shift_array(subvol_base, subvol_target, r_threshold=0.2)

                # Append these arrays to the round_shift and round_position storage
                round_shift[t, r] = shift
                round_position[t, r] = position

                # Save all shifts found thus far
                np.save(os.path.join(nbp_file.output_dir, 'round_shift.npy'), round_shift)
                np.save(os.path.join(nbp_file.output_dir, 'round_position.npy'), round_position)

            for c in use_channels:
                pbar.set_postfix({'tile': f'{t}', 'channel': f'{c}'})
                # Load in imaging npy volume
                target_image = load_tile(nbp_file, nbp_basic, t, r_mid, c)
                target_image = np.swapaxes(target_image, 0, 2)
                target_image = np.swapaxes(target_image, 1, 2)

                # Match histograms to unfiltered anchor and then sobel filter
                target_image = match_histograms(target_image, anchor_image_unfiltered)
                target_image = sobel(target_image)

                # next we split image into overlapping cuboids
                subvol_base, position = split_3d_image(image=anchor_image, z_subvolumes=z_subvols,
                                                       y_subvolumes=y_subvols, x_subvolumes=x_subvols,
                                                       z_box=z_box, y_box=y_box, x_box=x_box)
                subvol_target, _ = split_3d_image(image=target_image, z_subvolumes=z_subvols,
                                                  y_subvolumes=y_subvols, x_subvolumes=x_subvols,
                                                  z_box=z_box, y_box=y_box, x_box=x_box)

                # Find the subvolume shifts
                if c != c_ref:
                    shift = find_shift_array(subvol_base, subvol_target, r_threshold=0.2)
                else:
                    # We have already computed this set of shifts for c_ref
                    shift = round_shift[t, r_mid]

                # Append these arrays to the channel_shift and channel_position storage.
                # Reminder that these are not what will be used to directly compute the shift from (t, r_mid, c_ref) to
                # (t, r_mid, c) but these will rather be used to compute the transform from (t, r_ref, c_ref) to
                # (t, r_mid, c). Given that we know the transform from (t, r_ref, c_ref) to (t, r_mid, c_ref), the
                # desired transform can be easily computed

                channel_shift[t, c] = shift
                channel_position[t, c] = position

                # Save all shifts found thus far
                np.save(os.path.join(nbp_file.output_dir, 'channel_shift.npy'), channel_shift)
                np.save(os.path.join(nbp_file.output_dir, 'channel_position.npy'), channel_position)
            pbar.update(1)

    # Part 2: Compute the regression

    # For each tile, only need to do this for the n_rounds + channels not equal to ref channel
    with tqdm(total=len(use_tiles) * (n_rounds + len(use_channels) - 1)) as pbar:

        pbar.set_description(f"Computing regression for all transforms")

        for t in use_tiles:
            for r in use_rounds:
                pbar.set_postfix({'tile': f'{t}', 'round': f'{r}'})
                round_transform[t, r] = ols_regression_robust(round_shift[t, r], round_position[t, r], spread)
                pbar.update(1)
            for c in use_channels:
                pbar.set_postfix({'tile': f'{t}', 'channel': f'{c}'})
                aux_transform = ols_regression_robust(channel_shift[t, c], channel_position[t, c], spread)
                channel_transform[t, c] = compose_affine(aux_transform, invert_affine(round_transform[t, r_mid]))
                pbar.update(1)

        # Now regularise these and then combine these into total transforms
        round_transform[np.ix_(use_tiles, use_rounds)] = \
            regularise_transforms(round_transform[np.ix_(use_tiles, use_rounds)], 5, tile_origin.copy()[use_tiles])
        channel_transform[np.ix_(use_tiles, use_channels)] = \
            regularise_transforms(channel_transform[np.ix_(use_tiles, use_channels)], 5, tile_origin.copy()[use_tiles])

        for t in use_tiles:
            for r in use_rounds:
                for c in use_channels:
                    transform[t, r, c] = reformat_affine(compose_affine(channel_transform[t, c], round_transform[t, r]),
                                                         z_scale)

    # Part 3: ICP
    # This algorithm works well, but we need pixel level registration. For this reason, we use this transform as an
    # initial guess for an ICP algorithm

    for t in use_tiles:
        spot_yxz_ref[t] = spot_yxz(spot_details, t, nbp_basic.anchor_round, nbp_basic.anchor_channel,
                                   spot_no) * np.array([1, 1, z_scale])
        for r in use_rounds:
            for c in use_channels:
                spot_yxz_imaging[t, r, c] = spot_yxz(spot_details, t, r, c, spot_no) * np.array([1, 1, z_scale])
                if neighb_dist_thresh < 50:
                    # only keep isolated spots, those whose second neighbour is far away
                    isolated = get_isolated_points(spot_yxz_imaging[t, r, c], 2 * neighb_dist_thresh)
                    spot_yxz_imaging[t, r, c] = spot_yxz_imaging[t, r, c][isolated, :]

    # Don't import ICP as this is slow, instead, just run 50 iterations or so off each transform
    with tqdm(total=len(use_tiles) * len(use_rounds) * len(use_channels)) as pbar:

        pbar.set_description(f"Computing ICP for each [t,r,c]")
        for t in use_tiles:
            for r in use_rounds:
                for c in use_channels:
                    final_transform[t, r, c], n_matches[t, r, c], error[t, r, c] = icp(yxz_base=spot_yxz_ref[t],
                                                                                       yxz_target=spot_yxz_imaging[t, r, c],
                                                                                       dist_thresh=neighb_dist_thresh,
                                                                                       start_transform=transform[t, r, c],
                                                                                       n_iters=50, robust=False)
                    pbar.update(1)
    # Add convergence statistics to the debug notebook page.
    nbp_debug.n_matches = n_matches
    nbp_debug.error = error
    # These are not in use now but may be useful in the future
    nbp_debug.converged = None
    nbp_debug.failed = None
    nbp_debug.av_scaling = None
    nbp_debug.av_shifts = None
    nbp_debug.transform_outlier = None

    # add to register page of notebook
    nbp.start_transform = transform
    nbp.transform = final_transform
    nbp.round_shift = round_shift
    nbp.round_position = round_position
    nbp.round_transform = round_transform
    nbp.channel_shift = channel_shift
    nbp.channel_position = channel_position
    nbp.channel_transform = channel_transform

    return nbp, nbp_debug
