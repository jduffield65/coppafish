import warnings
import numpy as np
from tqdm import tqdm
from coppafish.stitch import compute_shift, update_shifts
from coppafish.find_spots import spot_yxz
from coppafish import utils
from coppafish.setup import Notebook, NotebookPage
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel


def register_cameras(nbp_basic: NotebookPage, nbp_file: NotebookPage, config: dict):
    """
    Script to align cameras, as the initial registration assumes these are all aligned. We find and return a shift
    from the shift camera to each other camera. Since the other registration finds the shift from the anchor cam to
    the shift cam + the true shift, adding these gives shift from anchor cam to all cams + true shift.

    Args:
        nbp_basic: basic info page of Notebook we're performing registration on (Notebook Page)
        nbp_file: file names page of Notebook we're performing registration on (Notebook Page)
        config: dictionary containing register_initial info (dict)

    Returns:
        shift: (n_tiles x n_rounds x n_channels x 3) array containing camera shifts from cam[shift_channel] to cam[c]
        for each round c in use. This is then just copied across n_tiles and n_rounds.
    """

    # Initialise recurring variables
    cam = nbp_basic.channel_camera
    laser = nbp_basic.channel_laser
    shift_channel = config['shift_channel']
    if shift_channel is None:
        shift_channel = nbp_basic.anchor_channel
    sample_tile = min(nbp_basic.use_tiles)
    sample_round = min([i for i in nbp_basic.use_rounds if i > 0])

    # Load and filter anchor image
    shift_channel_image = sobel(utils.nd2.get_raw_images(nbp_basic, nbp_file, [sample_tile], [sample_round],
                                                         [shift_channel], list(np.arange(nbp_basic.nz // 2 - 10,
                                                                                         nbp_basic.nz // 2 + 10)))[0, 0, 0])
    # Initialise the variable that we will return
    shift = np.zeros((nbp_basic.n_channels, 3), dtype=int)

    # The body of the code starts here. We will be finding the shifts taking cam[shift_channel] to cam[c] for all c in
    # use.

    # Create a list of cameras that are in use, get rid of duplicates and as we're finding shits from shift_channel_cam
    # to all other cams, don't need to include shift_cam
    use_cams = list(set([cam[i] for i in nbp_basic.use_channels]).symmetric_difference(set([cam[shift_channel]])))

    # Now we will pick a representative channel for each camera:
    # Create an empty list for the sample channels, we'll populate this in the next for loop
    sample_channels = []
    # For each camera we choose a channel which will give us a sample image.
    # According to ground truth data, the best images are those where laser and camera freqs are similar
    for i in range(len(use_cams)):
        # potential_channels is just the preimage of the camera function intersected with use_channels
        potential_channels = [j for j in nbp_basic.use_channels if cam[j] == use_cams[i]]
        # The channel we choose will minimise |laser_freq - camera_freq|
        delta_freq = [abs(cam[j]-laser[j]) for j in potential_channels]
        sample_channels.append(potential_channels[np.argmin(delta_freq)])

    # Load in all sample volumes to an array called sample_image. Sample_image[i] refers to the camera use_cams[i]
    sample_image = utils.nd2.get_raw_images(nbp_basic, nbp_file, [sample_tile], [sample_round], sample_channels,
                                            list(np.arange(nbp_basic.nz//2 - 10, nbp_basic.nz//2 + 10)))[0, 0]

    # Initialise filtered_sample_image
    filtered_sample_image = np.zeros(sample_image.shape)
    # Filter each image with a Sobel filter to improve registration
    for i in range(sample_image.shape[0]):
        filtered_sample_image[i] = sobel(sample_image[i])
    # Delete sample_image to save memory
    del sample_image

    # Store the cam_shifts in this array, which will then be used to populate cam shifts by channel
    cam_shift = np.zeros((len(use_cams), 3), dtype=int)

    # Now for each sample image, detect the shift taking cam[shift_channel] to cam[c]
    for i in range(filtered_sample_image.shape[0]):
        cam_shift[i], _, _ = phase_cross_correlation(filtered_sample_image[i], shift_channel_image)

    # Final part is to populate the shift array. Loop through all channels in use and update them accordingly
    for c in nbp_basic.use_channels:
        # Nothing to do for channels from shift_channel_cam, as there is no shift to itself
        if cam[c] != cam[shift_channel]:
            # First get the index of the camera of channel c, as listed in use_cams
            camera_index = use_cams.index(cam[c])
            # Now to find the shift from anchor cam to channel c cam, we add the shift from anchor cam to shift cam, and
            # then from shift cam to channel c cam
            shift[c] = cam_shift[camera_index]

    # Reformat shift array from n_channels x 3 to n_tiles x n_rounds x n_channels x 3 for consistency with other code
    # First copy the array for each round
    shift = np.repeat(shift[np.newaxis, :, :], nbp_basic.n_rounds, axis=0)
    # Now copy the array for each tile
    shift = np.repeat(shift[np.newaxis, :, :, :], nbp_basic.n_tiles, axis=0)

    return shift


def register_initial_scale(nbp_basic: NotebookPage, nbp_file: NotebookPage):
    """
    Script to find initial estimate of z-shifts across rounds. The samples seem to expand a lot throughout the rounds
    and when the initial scale is far from the true scale the ICP algorithm will not find it.

    Args:
        nbp_basic: basic info page of Notebook we're performing registration on (Notebook Page)
        nbp_file: file names page of Notebook we're performing registration on (Notebook Page)
        config: dictionary containing register_initial info (dict)

    Returns:
        z_scale: (n_rounds) array of the z_scale which should be applied to the anchor to give the reference
        image. Expect this to be < 1 for low rounds and then increase to 1 as rounds approach anchor round.
    """
    # Algorithm works by loading in images for central tile t and reference channel c_ref for all rounds including the
    # reference round r_ref. Then we progress by the following algorithm:
    # 1.) find the shift taking the top half z-planes of [t, r_ref] to the top half z-planes of [t,r]
    # 2.) find the shift taking the bottom half z-planes of [t, r_ref] to the bottom half z-planes of [t,r]
    # The difference between the z-coordinates of these shifts will allow us to approximate the z-scale.

    # Initialise commonly used variables
    r_ref, c_ref = nbp_basic.anchor_round, nbp_basic.anchor_channel
    use_tiles, use_rounds, use_z = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_z
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]    # Tile positions of tiles in use
    tilepos_yx_centred = tilepos_yx - np.mean(nbp_basic.tilepos_yx, axis=0)     # Tile positions with respect to centre
    central_tile = np.argmin(np.linalg.norm(tilepos_yx_centred, axis=1))
    tile_sz = nbp_basic.tile_sz
    yx_centre = tile_sz // 2

    # Load in all volumes to an array called image. image[r] refers to the raw image for round r, channel c_ref.
    image = utils.nd2.get_raw_images(nbp_basic, nbp_file, [central_tile], use_rounds, [c_ref], use_z)[0, :, 0,
            yx_centre-tile_sz//4:yx_centre+tile_sz//4, yx_centre-tile_sz//4:yx_centre+tile_sz//4]

    filtered_image = np.zeros(image.shape)
    # Filter each image with a Sobel filter to improve registration
    for i in range(image.shape[0]):
        filtered_image[i] = sobel(image[i])
    # Load and filter the anchor image too
    anchor_image = sobel(utils.nd2.get_raw_images(nbp_basic, nbp_file, [central_tile], [r_ref], [c_ref], use_z)[0, 0, 0,
                         yx_centre-tile_sz//4:yx_centre+tile_sz//4, yx_centre-tile_sz//4:yx_centre+tile_sz//4])

    # Now we find the shifts taking the top 1/3 of anchor_image to the top 1/3 of image[r] for each r and likewise
    # for the bottom 1/3.
    lower_shift, upper_shift = np.zeros((len(use_rounds), 3)), np.zeros((len(use_rounds), 3))
    # loop through all rounds and fill in the respective shifts
    for r in range(len(use_rounds)):
        lower_shift[r], _, _ = phase_cross_correlation(filtered_image[r, :, :, :len(use_z)//3],
                                                       anchor_image[:, :, :len(use_z)//3])
        upper_shift[r], _, _ = phase_cross_correlation(filtered_image[r, :, :, 2 * len(use_z) // 3:],
                                                       anchor_image[:, :, 2 * len(use_z) // 3:])

    # Now assume that these shifts have perfectly aligned the middle of lower regions to the middle of the anchor's
    # lower region (haha) and likewise with the upper regions. Then the ratio of the new length of this interval
    # compared to the starting length of this interval (2/3 * num_z) approximates the scaling.
    # Name the scaling variable z_expansion to avoid confusion with z_scale used to convert z pixels to same units as
    # yx pixels.
    len0 = 2 * len(use_z) // 3
    z_expansion = 1 + (upper_shift[:, 2] - lower_shift[:, 2])/len0

    return z_expansion


def register_initial(config: dict, nbp_basic: NotebookPage, nbp_file: NotebookPage, spot_details: np.ndarray,
                     spot_no: np.ndarray) -> NotebookPage:
    """
    This finds the shift between ref round/channel to each imaging round for each tile.
    These are then used as the starting point for determining the affine transforms in `pipeline/register.py`.

    See `'register_initial'` section of `notebook_comments.json` file
    for description of the variables in the page.

    Args:
        config: Dictionary obtained from `'register_initial'` section of config file.
        nbp_basic: `basic_info` notebook page
        nbp_file: 'file_names' notebook page
        spot_details: `int [n_spots x 3]`.
            `spot_details[s]` is `[ y, x, z]` of spot `s`.
            This is saved in the find_spots notebook page i.e. `nb.find_spots.spot_details`.
        spot_no: 'int[n_tiles x n_rounds x n_channels]'
            'spot_no[t,r,c]' is num_spots found on that [t,r,c]
            This is saved on find_spots notebook page

    Returns:
        `NotebookPage[register_initial]` - Page contains information about how shift between ref round/channel
            to each imaging round for each tile was found.
    """
    nbp_debug = NotebookPage("register_initial")
    if config['shift_channel'] is None:
        config['shift_channel'] = nbp_basic.ref_channel
    if not np.isin(config['shift_channel'], nbp_basic.use_channels):
        raise ValueError(f"config['shift_channel'] should be in nb.basic_info.use_channels, but value given is\n"
                         f"{config['shift_channel']} which is not in use_channels = {nbp_basic.use_channels}.")
    nbp_debug.shift_channel = config['shift_channel']

    coords = ['y', 'x', 'z']
    shifts = [{}]
    start_shift_search = np.zeros((nbp_basic.n_rounds, 3, 3), dtype=int)
    for i in range(len(coords)):
        shifts[0][coords[i]] = np.arange(config['shift_min'][i],
                                         config['shift_max'][i] +
                                         config['shift_step'][i] / 2, config['shift_step'][i]).astype(int)
        start_shift_search[nbp_basic.use_rounds, i, :] = [config['shift_min'][i], config['shift_max'][i],
                                                          config['shift_step'][i]]
    if not nbp_basic.is_3d:
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0
        shifts[0]['z'] = np.array([0], dtype=int)
        start_shift_search[:, 2, :2] = 0
    shifts = shifts * nbp_basic.n_rounds  # get one set of shifts for each round

    shift = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds, 3), dtype=int)
    shift_score = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds), dtype=float)
    shift_score_thresh = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds), dtype=float)

    c_ref = nbp_basic.ref_channel
    r_ref = nbp_basic.ref_round
    c_imaging = config['shift_channel']
    # to convert z coordinate units to xy pixels when calculating distance to nearest neighbours
    z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy

    # Load in initial z_expansion factors first. This is important as scaling and translation do not commute so we need
    # to scale first to keep consistent with the convention in the ICP step.
    z_expansion = register_initial_scale(nbp_basic, nbp_file)

    with tqdm(total=len(nbp_basic.use_rounds) * len(nbp_basic.use_tiles)) as pbar:
        pbar.set_description(f"Finding shift from ref_round({r_ref})/ref_channel({c_ref}) to channel {c_imaging} "
                             f"of all imaging rounds")

        for r in nbp_basic.use_rounds:
            for t in nbp_basic.use_tiles:
                pbar.set_postfix({'round': r, 'tile': t})
                shift[t, r], shift_score[t, r], shift_score_thresh[t, r] = \
                    compute_shift(spot_yxz(spot_details, t, r_ref, c_ref, spot_no) * [1, 1, z_expansion[r]],
                                  spot_yxz(spot_details, t, r, c_imaging, spot_no),
                                  config['shift_score_thresh'], config['shift_score_thresh_multiplier'],
                                  config['shift_score_thresh_min_dist'], config['shift_score_thresh_max_dist'],
                                  config['neighb_dist_thresh'], shifts[r]['y'], shifts[r]['x'], shifts[r]['z'],
                                  config['shift_widen'], config['shift_max_range'], z_scale,
                                  config['nz_collapse'], config['shift_step'][2])[:3]
                good_shifts = shift_score[:, r] > shift_score_thresh[:, r]
                if np.sum(good_shifts) >= 3:
                    # once found shifts, refine shifts to be searched around these
                    for i in range(len(coords)):
                        shifts[r][coords[i]] = update_shifts(shifts[r][coords[i]], shift[good_shifts, r, i])
                pbar.update(1)
    pbar.close()

    # amend shifts for which score fell below score_thresh
    shift_outlier = shift.copy()
    shift_score_outlier = shift_score.copy()
    n_shifts = len(nbp_basic.use_tiles)
    final_shift_search = np.zeros_like(start_shift_search)
    final_shift_search[:, :, 2] = start_shift_search[:, :, 2]  # spacing does not change
    for r in nbp_basic.use_rounds:
        good_shifts = shift_score[:, r] > shift_score_thresh[:, r]
        for i in range(len(coords)):
            # change shift search to be near good shifts found
            # this will only do something if 3>sum(good_shifts)>0, otherwise will have been done in previous loop.
            if np.sum(good_shifts) > 0:
                shifts[r][coords[i]] = update_shifts(shifts[r][coords[i]], shift[good_shifts, r, i])
            elif good_shifts.size > 0:
                shifts[r][coords[i]] = update_shifts(shifts[r][coords[i]], shift[:, r, i])
        final_shift_search[r, :, 0] = [np.min(shifts[r][key]) for key in shifts[r].keys()]
        final_shift_search[r, :, 1] = [np.max(shifts[r][key]) for key in shifts[r].keys()]
        shift_outlier[good_shifts, r] = 0  # only keep outlier information for not good shifts
        shift_score_outlier[good_shifts, r] = 0
        if (np.sum(good_shifts) < 2 and n_shifts > 4) or (np.sum(good_shifts) == 0 and n_shifts > 0):
            warnings.warn(f"Round {r}: {n_shifts - np.sum(good_shifts)}/{n_shifts} "
                          f"of shifts fell below score threshold")
        for t in np.where(good_shifts == False)[0]:
            if t not in nbp_basic.use_tiles:
                continue
            # re-find shifts that fell below threshold by only looking at shifts near to others found
            # score set to 0 so will find do refined search no matter what.
            shift[t, r], shift_score[t, r] = compute_shift(spot_yxz(spot_details, t, r_ref, c_ref, spot_no),
                                                           spot_yxz(spot_details, t, r, c_imaging, spot_no), 0, None,
                                                           None, None, config['neighb_dist_thresh'], shifts[r]['y'],
                                                           shifts[r]['x'], shifts[r]['z'], None, None, z_scale,
                                                           config['nz_collapse'], config['shift_step'][2])[:2]
            warnings.warn(f"\nShift for tile {t} to round {r} changed from\n"
                          f"{shift_outlier[t, r]} to {shift[t, r]}.")

    # Reformat the shifts into an array of size (n_tiles x n_rounds x n_channels x 3) by copying the array across
    # each channel in use
    reformatted_shift = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds, nbp_basic.n_channels, 3), dtype=int)
    for c in nbp_basic.use_channels:
        reformatted_shift[:, :, c, :] = shift

    cam_shift = np.zeros(reformatted_shift.shape)
    # Now if we're in a quad-cam experiment, compute the camera shifts
    if nbp_basic.channel_camera is not None:
        cam_shift = register_cameras(nbp_basic, nbp_file, config)

    nbp_debug.cam_shift = cam_shift
    nbp_debug.shift = reformatted_shift + cam_shift
    nbp_debug.z_expansion_factor = z_expansion
    nbp_debug.start_shift_search = start_shift_search
    nbp_debug.final_shift_search = final_shift_search
    nbp_debug.shift_score = shift_score
    nbp_debug.shift_score_thresh = shift_score_thresh
    nbp_debug.shift_outlier = shift_outlier
    nbp_debug.shift_score_outlier = shift_score_outlier

    return nbp_debug


# nb = Notebook('C:/Users\Reilly\Downloads\wetransfer_e-2210-001_cp_settings-ini_2022-10-31_1542/notebook.npz',
#               'C:/Users\Reilly\Downloads\wetransfer_e-2210-001_cp_settings-ini_2022-10-31_1542/E-2210-001_CP_settings.ini')
# z_expansion = register_initial_scale(nb.basic_info, nb.file_names)