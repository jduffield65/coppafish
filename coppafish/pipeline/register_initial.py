import numpy as np
from tqdm import tqdm
from coppafish.stitch import compute_shift, update_shifts
from coppafish.find_spots import spot_yxz
import warnings
from coppafish.setup import Notebook, NotebookPage
from coppafish.plot.raw import get_raw_images
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel
from scipy.fft import fft2, fftshift
from skimage.transform import warp_polar


def register_initial(config: dict, nbp_basic: NotebookPage, spot_details: np.ndarray, spot_no: np.ndarray) \
        -> NotebookPage:
    """
    This finds the shift between ref round/channel to each imaging round for each tile.
    These are then used as the starting point for determining the affine transforms in `pipeline/register.py`.

    See `'register_initial'` section of `notebook_comments.json` file
    for description of the variables in the page.

    Args:
        config: Dictionary obtained from `'register_initial'` section of config file.
        nbp_basic: `basic_info` notebook page
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
    with tqdm(total=len(nbp_basic.use_rounds) * len(nbp_basic.use_tiles)) as pbar:
        pbar.set_description(f"Finding shift from ref_round({r_ref})/ref_channel({c_ref}) to channel {c_imaging} "
                             f"of all imaging rounds")

        for r in nbp_basic.use_rounds:
            for t in nbp_basic.use_tiles:
                pbar.set_postfix({'round': r, 'tile': t})
                shift[t, r], shift_score[t, r], shift_score_thresh[t, r] = \
                    compute_shift(spot_yxz(spot_details, t, r_ref, c_ref, spot_no),
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

    nbp_debug.shift = shift
    nbp_debug.start_shift_search = start_shift_search
    nbp_debug.final_shift_search = final_shift_search
    nbp_debug.shift_score = shift_score
    nbp_debug.shift_score_thresh = shift_score_thresh
    nbp_debug.shift_outlier = shift_outlier
    nbp_debug.shift_score_outlier = shift_score_outlier

    return nbp_debug


def register_cameras(nb: Notebook, t: int) -> np.ndarray:
    """
    Initial shifts are computed based on a single shift_channel. This channel can be found in
    nb.register_initial.shift_channel. When the cameras are offset from one another, the initial shift should be
    different for channels inherited from different cameras than that belonging to the shift_channel.
    For this reason, we first align cameras.

    Args:
        nb: Notebook we're rperformeing registration on
        t: tile we'd like to compute this registration on.

    Returns:
        shifts: The shifts from each camera to the reference channel camera (np.ndarray (3 x 3))
        angles: The yx acw rotation from each camera to the reference channel camera (np.ndarray ( 1 x 3))
    """

    # Load in basic_info page
    nbp_basic = nb.basic_info
    # First check there is indeed an anchor channel
    if not nbp_basic.anchor_channel:
        raise Exception('No anchor round to align to!')

    # Now that we've established these exist, lets define the anchor round and channel as variables
    anchor_round = nbp_basic.anchor_round
    anchor_channel = nbp_basic.anchor_channel

    # Now, let us list the channels and lasers associated with each camera
    blue_camera_channels = [i for i in range(len(nbp_basic.channel_camera)) if nbp_basic.channel_camera[i] == 405]
    blue_lasers = [nbp_basic.channel_laser[i] for i in blue_camera_channels]
    green_camera_channels = [i for i in range(len(nbp_basic.channel_camera)) if nbp_basic.channel_camera[i] == 470]
    green_lasers = [nbp_basic.channel_laser[i] for i in green_camera_channels]
    orange_camera_channels = [i for i in range(len(nbp_basic.channel_camera)) if nbp_basic.channel_camera[i] == 555]
    orange_lasers = [nbp_basic.channel_laser[i] for i in orange_camera_channels]
    red_camera_channels = [i for i in range(len(nbp_basic.channel_camera)) if nbp_basic.channel_camera[i] == 640]
    red_lasers = [nbp_basic.channel_laser[i] for i in red_camera_channels]

    # Define a cameras list showing which cameras are in use
    index = 0
    cameras = []
    anchor_cam = []
    # For each camera, the corresponding channel will be stored in sample_channels
    sample_channels = []
    # Next, we know from ground truth data that good camera channels all have the same excitation and camera freq. If
    # these channels are contained, use them. Else, use the first that shows up.
    if red_camera_channels:
        cameras.append(index)
        if 640 in red_lasers:
            sample_channels.append(red_camera_channels[red_lasers.index(640)])
        else:
            sample_channels.append(red_camera_channels[0])
        if anchor_channel in red_camera_channels:
            anchor_cam = index
        index = index + 1
    if blue_camera_channels:
        cameras.append(index)
        if 405 in blue_lasers:
            sample_channels.append(blue_camera_channels[blue_lasers.index(405)])
        else:
            sample_channels.append(blue_camera_channels[0])
        if anchor_channel in blue_camera_channels:
            anchor_cam = index
        index = index + 1
    if green_camera_channels:
        cameras.append(index)
        if 470 in green_lasers:
            sample_channels.append(green_camera_channels[green_lasers.index(470)])
        else:
            sample_channels.append(green_camera_channels[0])
        if anchor_channel in green_camera_channels:
            anchor_cam = index
        index = index + 1
    if orange_camera_channels:
        cameras.append(index)
        if 555 in orange_lasers:
            sample_channels.append(orange_camera_channels[orange_lasers.index(555)])
        else:
            sample_channels.append(orange_camera_channels[0])
        if anchor_channel in orange_camera_channels:
            anchor_cam = index

    # Now define the non_anchor channels and cameras
    non_anchor_channels = [i for i in sample_channels if i != sample_channels[anchor_cam]]
    non_anchor_cam = [i for i in cameras if i != anchor_cam]

    # Now that we have a channel for each camera, we'd like to load in a volume corresponding to each of those channels.
    # Round 0 is always a bit dodgy. So let's use the first nonzero round
    if len(nbp_basic.use_rounds) == 1:
        raise Exception('Need more than just the reference round!')

    sample_round = min([i for i in nbp_basic.use_rounds if i > 0])

    # Now look at all the cameras in use and load in the middle of the sample images
    sample_image = get_raw_images(nb, [t], [sample_round], non_anchor_channels,
                                  list(np.arange(nbp_basic.nz//2 - 10, nbp_basic.nz//2 + 10)))[0, 0]
    anchor_image = get_raw_images(nb, [t], [sample_round], [anchor_channel],
                                  list(np.arange(nbp_basic.nz//2 - 10, nbp_basic.nz//2 + 10)))[0, 0, 0]

    # Initialise filtered_sample_image
    filtered_sample_image = np.zeros(sample_image.shape)

    # Filter each image with a Sobel filter to improve registration
    for i in range(sample_image.shape[0]):
        filtered_sample_image[i] = sobel(sample_image[i])

    anchor_image = sobel(anchor_image)

    # Now create an array to store all the shifts to the anchor camera.
    cam_shift = np.zeros((len(cameras), 3), dtype=int)
    error = np.zeros(len(cameras))
    phase_diff = np.zeros(len(cameras))
    angle = np.zeros(len(cameras))

    for i in non_anchor_cam:
        cam_shift[i], error[i], phase_diff[i] = phase_cross_correlation(anchor_image, filtered_sample_image[i])
        angle[i] = detect_rotation(anchor_image[:, :, 10], filtered_sample_image[i, :, :, 10])

    return cam_shift, error, angle


def detect_rotation(ref: np.ndarray, extra: np.ndarray):
    """
    Function which takes in 2 2D images which are rotated and translated with respect to one another and returns the
    rotation angle in degrees between them.
    Args:
        ref: reference image
        extra: moving image
    Returns:
        angle: Anticlockwise angle which, upon application to ref, yields extra.
    """
    # work with shifted FFT log-magnitudes
    ref_ft = np.abs(fftshift(fft2(ref)))
    extra_ft = np.abs(fftshift(fft2(extra)))

    # Create log-polar transformed FFT mag images and register
    shape = ref_ft.shape
    radius = shape[0] // 2  # only take lower frequencies
    warped_ref_ft = warp_polar(ref_ft, radius=radius, scaling='log')
    warped_extra_ft = warp_polar(extra_ft, radius=radius, scaling='log')

    warped_ref_ft = warped_ref_ft[:shape[0] // 2, :]  # only use half of FFT
    warped_extra_ft = warped_extra_ft[:shape[0] // 2, :]
    warped_ref_ft[np.isnan(warped_extra_ft)] = 0
    warped_extra_ft[np.isnan(warped_extra_ft)] = 0

    shifts = phase_cross_correlation(warped_ref_ft, warped_extra_ft, upsample_factor=100,
                                     reference_mask=np.ones(warped_ref_ft.shape, dtype=int) - np.isnan(warped_ref_ft),
                                     moving_mask=np.ones(warped_ref_ft.shape, dtype=int) - np.isnan(warped_extra_ft),
                                     normalization=None)

    # Use translation parameters to calculate rotation parameter
    shift_angle = shifts[0]

    return shift_angle


nb1 = Notebook('C:/Users/Reilly/Desktop/Sample Notebooks/Izzie/notebook_single_tile.npz')
cam_shift, error, angle = register_cameras(nb1, 55)

print("Hello World")