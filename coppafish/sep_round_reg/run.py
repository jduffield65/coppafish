from typing import List, Tuple, Optional
from coppafish.pipeline.run import initialize_nb, run_extract, run_find_spots, run_stitch
from coppafish.sep_round_reg import base
import numpy as np
import scipy.ndimage as snd
from skimage.transform import rotate
from skimage.filters import window
from skimage.registration import phase_cross_correlation
from skimage.exposure import equalize_hist
from skimage.color import rgb2gray
import matplotlib
import matplotlib.pyplot as plt
from skimage import data

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp
matplotlib.use('Qt5Agg')
matplotlib.pyplot.style.use('dark_background')


def run_sep_round_reg(config_file: str, config_file_full: str, channels_to_save: List,
                      transform: Optional[np.ndarray] = None):
    """
    This runs the pipeline for a separate round up until the end of the stitching stage and then finds the
    affine transform that takes it to the anchor image of the full pipeline run.
    It then saves the corresponding transformed images for the channels of the separate round indicated by
    `channels_to_save`.

    Args:
        config_file: Path to config file for separate round.
            This should have only 1 round, that round being an anchor round (though not the anchor channel from the
            initial experiment) and only one channel being used so filtering is only done on the anchor channel.
        config_file_full: Path to config file for full pipeline run, for which full notebook exists.
        channels_to_save: Channels of the separate round, that will be saved to the output directory in
            the same coordinate system as the anchor round of the full run.
        transform: `float [4 x 3]`.
            Can provide the affine transform which transforms the separate round onto the anchor
            image of the full pipeline run. If not provided, it will be computed.
    """
    # Get all information from full pipeline results - global spot positions and z scaling

    # Full notebook for the experiment we've already run
    nb_full = initialize_nb(config_file_full)

    # run pipeline to get as far as a stitched DAPI image for the separate round
    nb_sep = initialize_nb(config_file)
    run_extract(nb_sep)
    run_find_spots(nb_sep)
    run_stitch(nb_sep)

    # Import both DAPI's from both notebooks:
    # Import dapi from full notebook
    dapi_full_dir = nb_full.file_names.big_dapi_image
    dapi_full_read = np.load(dapi_full_dir)
    dapi_full = dapi_full_read.f.arr_0

    # Import dapi from sep round notebook
    dapi_sep_dir = nb_sep.file_names.big_dapi_image
    dapi_sep_read = np.load(dapi_sep_dir)
    dapi_sep = dapi_sep_read.f.arr_0

    # Now that we have the dapi image for both we can begin registration on these images. Look at central z-planes for
    # of registration (parts 1 and 2)
    rigid_transform(dapi_full, dapi_sep)


def register_manual_shift(target_image: np.ndarray, offset_image: np.ndarray):
    # Target and offset are both 3D volumes but detection only uses mid z_plane
    # Manually find shift between mid z_planes
    target_image_mid = target_image[target_image.shape[0] // 2]
    offset_image_mid = offset_image[offset_image.shape[0] // 2]
    # Normalise middle z-planes
    target_image_mid = target_image_mid / np.max(target_image_mid)
    offset_image_mid = offset_image_mid / np.max(offset_image_mid)
    # open napari interface and allow user to input data
    initial_shift, ref_points_target, ref_points_offset = base.manual_shift2(target_image_mid, offset_image_mid)
    ref_points_target = np.array(ref_points_target, dtype='int')
    ref_points_offset = np.array(ref_points_offset, dtype='int')
    # Convert initial shift into 3D
    initial_shift = np.insert(initial_shift, 0, 0)
    # Apply this shift using scipy ndimage package
    offset_image = snd.shift(offset_image, -initial_shift)
    return offset_image, ref_points_target, ref_points_offset


def register_detect_rotation(target_image: np.ndarray, offset_image: np.ndarray, ref_points_target, ref_points_offset):
    # since input images are 3D volumes, we must take the mid z_plane
    target_image_mid = target_image[target_image.shape[0] // 2]
    offset_image_mid = offset_image[offset_image.shape[0] // 2]

    # Radius is determined to be the min distance from both centroids to their respective boundaries
    centroid_target = np.mean(ref_points_target, axis=0, dtype='int')
    centroid_offset = np.mean(ref_points_offset, axis=0, dtype='int')
    # Dimensions are same for both images
    image_dims = np.array(target_image_mid.shape, dtype=int)
    # Now we choose the biggest circle fitting in full target, biggest circle fitting in offset, then take smaller one
    radius_target = np.min([centroid_target, image_dims - centroid_target])
    radius_offset = np.min([centroid_offset, image_dims - centroid_offset])
    radius = int(np.min([radius_target, radius_offset]))
    # Now crop the images around this centre with this min radius
    target_image_mid = target_image_mid[centroid_target[0] - radius:centroid_target[0] + radius,
                       centroid_target[1] - radius:centroid_target[1] + radius]
    offset_image_mid = offset_image_mid[centroid_offset[0] - radius:centroid_offset[0] + radius,
                       centroid_offset[1] - radius:centroid_offset[1] + radius]

    # Now apply the rotation detection algorithm on these. As these are 2 DAPIs, we shouldn't need
    # to do any processing on them. The only thing we should do is window them.
    # Apply Hann Window to Images 1 and 2
    target_image_mid = target_image_mid * (window('hann', target_image_mid.shape) ** 0.1)
    offset_image_mid = offset_image_mid * (window('hann', offset_image_mid.shape) ** 0.1)
    angle, error = base.detect_rotation(target_image_mid, offset_image_mid)

    # rotate each z-plane by this amount
    for z in range(offset_image.shape[0]):
        offset_image[z] = rotate(offset_image[z], -angle)

    # This normalises offset_image so we must also normalise target_image
    target_image = target_image/np.max(target_image)

    return angle, error, target_image, offset_image, radius


def register_detect_rotation2(target_image: np.ndarray, offset_image: np.ndarray, ref_points_target, ref_points_offset):
    # since input images are 3D volumes, we must take the mid z_plane
    target_image_mid = target_image[target_image.shape[0] // 2]
    offset_image_mid = offset_image[offset_image.shape[0] // 2]

    # Dimensions are same for both images
    image_dims = np.array(target_image_mid.shape, dtype=int)
    # Now we choose the biggest circle fitting in full target, biggest circle fitting in offset, then take smaller one
    radius = 250
    angle_vec = np.zeros(ref_points_target.shape[0])
    # Now we'll detect rotations at each of our ref_points
    for i in range(ref_points_target.shape[0]):
        target_sample = target_image_mid[ref_points_target[i, 0]-radius: ref_points_target[i, 0]+radius,
                        ref_points_target[i, 1]-radius: ref_points_target[i, 1]+radius] * \
                        (window('hann', [2*radius, 2*radius]) ** 0.1)
        offset_sample = offset_image_mid[ref_points_offset[i, 0] - radius: ref_points_offset[i, 0] + radius,
                        ref_points_offset[i, 1] - radius: ref_points_offset[i, 1] + radius] * \
                        (window('hann', [2 * radius, 2 * radius]) ** 0.1)
        angle_vec[i], error = base.detect_rotation(target_sample, offset_sample)

    angle = np.mean(angle_vec)

    # rotate each z-plane by this amount
    for z in range(offset_image.shape[0]):
        offset_image[z] = rotate(offset_image[z], -angle)

    # This normalises offset_image so we must also normalise target_image
    target_image = target_image/np.max(target_image)

    return angle, error, target_image, offset_image, radius


def rigid_transform(target_image: np.ndarray, offset_image: np.ndarray):
    # target_image and offset_image will be cropped, mid 20% of z planes
    # 1.) Have the user manually select an initial shift in yx, apply this shift to the sep round
    # 2.) Detect the rotation between the first and second image, apply this rotation to the sep round
    # 3.) Use phase correlation to find optimal shift in 3D between two images and apply to sep round

    # Start by cropping z- planes
    #  We'll only need about 10% of the volume around the middle
    z_range = range(int(0.45 * target_image.shape[0]), int(0.55 * target_image.shape[0]))
    target_image = target_image[z_range]
    offset_image = offset_image[z_range]

    # Rescale so max = 1
    target_image = target_image/np.max(target_image)
    offset_image = offset_image/np.max(offset_image)

    # Step 1.)
    # Manual Shift
    offset_image, ref_points_target, ref_points_offset = register_manual_shift(target_image, offset_image)

    # Step 2.)
    # Detect rotation between DAPI iamges. We take the biggest circle centred around the centre of the three points
    # chosen in step 1
    angle, error, target_image, offset_image, radius = register_detect_rotation2(target_image, offset_image,
                                                                                ref_points_target, ref_points_offset)

    # Step 3:
    # 3D shift detection.
    centroid_target = np.mean(ref_points_target, axis=0, dtype='int')
    centroid_offset = np.mean(ref_points_offset, axis=0, dtype='int')

    shift, error, phase = phase_cross_correlation(target_image[:, centroid_target[0]-radius:centroid_target[0]+radius,
                                           centroid_target[1]-radius:centroid_target[1]+radius],
                                           offset_image[:, centroid_offset[0]-radius:centroid_offset[0]+radius,
                                           centroid_offset[1]-radius:centroid_offset[1]+radius], upsample_factor=1,
                                           normalization=None)
    
    # commenting out this shift as it doesn't seem very good
    # offset_image = snd.shift(offset_image, shift)
    rgb_overlay = np.zeros((4090, 5967, 3))
    rgb_overlay[:, :, 0] = target_image[3, :4090, :5967]
    rgb_overlay[:, :, 2] = offset_image[3, :4090, :5967]
    plt.imshow(rgb_overlay)

    return offset_image, target_image, shift, angle, error


dapi_partial = np.load('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Sep Round/Sep/dapi_image.npz')
dapi_partial = dapi_partial.f.arr_0
dapi_full = np.load('C:/Users/Reilly/Desktop/Sample Notebooks/Christina/Sep Round/Full/dapi_image.npz')
dapi_full = dapi_full.f.arr_0
rigid_transform(target_image=dapi_full, offset_image=dapi_partial)
# print(shift, angle, error)
# astro = rgb2gray(data.astronaut())
# astro_new = snd.shift(astro, [10, 20])
# astro_new = rotate(astro_new, 7)
#
# base.manual_shift(astro, astro_new)
