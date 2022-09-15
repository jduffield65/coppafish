from typing import List, Tuple, Optional
from coppafish.register import get_single_affine_transform
from coppafish.pipeline.run import initialize_nb, run_extract, run_find_spots, run_stitch
from coppafish import setup, utils, Notebook
from coppafish.call_spots import get_non_duplicate
from coppafish.stitch import compute_shift
from coppafish.find_spots import get_isolated_points
from coppafish.spot_colors import apply_transform
from coppafish.plot.register.shift import view_shifts
from ..sep_round_reg import base
import numpy as np
import scipy.ndimage as snd
from skimage.transform import rotate
from skimage.filters import window
from skimage.registration import phase_cross_correlation
import os
import warnings
import matplotlib
import matplotlib.pyplot as plt

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

    # run pipeline to get as far as a DAPI image for the separate round
    nb_sep = initialize_nb(config_file)
    run_extract(nb_sep)
    run_find_spots(nb_sep)
    run_stitch(nb_sep)

    # Number of z planes is roughly the same in both nb sep and nb full
    num_z = nb_full.basic_info.use_z

    # Import both DAPI's from both notebooks:
    # Import dapi from full notebook
    dapi_full_dir = nb_full.file_names.big_dapi_image
    dapi_full_read = np.load(dapi_full_dir)
    dapi_full = dapi_full_read.f.arr_0

    # Import dapi from sep round notebook
    dapi_sep_dir = nb_sep.file_names.big_dapi_image
    dapi_sep_read = np.load(dapi_sep_dir)
    dapi_sep = dapi_sep_read.f.arr_0

    # Take mid z-planes as first 2 registration steps are done in 2D
    dapi_full_mid = dapi_full[int(num_z / 2)]
    dapi_sep_mid = dapi_sep[int(num_z/2)]

    # Now that we have the dapi image for both we can begin registration on these images. Look at central z-planes.
    # We'll split this up into 3 steps:
    # 1.) Have the user manually select an initial shift, apply this shift to the sep round
    # 2.) Detect the rotation between the first and second image, apply this rotation to the sep round
    # 3.) Use phase correlation to find optimal shift in 3D between two images and apply to sep round

    # Step 1.)
    # Manual Shift
    dapi_sep, ref_points_full, ref_points_sep = register_manual_shift(dapi_full, dapi_sep)

    # Step 2.)
    # Detect rotation between DAPI iamges. We take the biggest circle centred around the centre of the three points
    # chosen in step 1
    dapi_full_cropped, dapi_sep_cropped = register_detect_rotation(dapi_full, dapi_sep, ref_points_full, ref_points_sep)

    # Step 3:
    # 3D shift detection
    shift, error = phase_cross_correlation(dapi_full_cropped, dapi_sep_cropped, upsample_factor=1, normalization=None)
    dapi_sep = snd.shift(dapi_sep, shift)


def register_manual_shift(target_image: np.ndarray, offset_image: np.ndarray):
    # Target and offset are both 3D volumes but detection only uses mid z_plane
    num_z = target_image.shape[0]
    # Manually find shift between mid z_planes
    target_image_mid = target_image[int(num_z/2)]
    offset_image_mid = offset_image[int(num_z / 2)]
    initial_shift, ref_points_target, ref_points_offset = base.manual_shift(target_image_mid, offset_image_mid)
    # Convert initial shift into 3D
    initial_shift = np.insert(initial_shift, 0, 0)
    # Apply this shift using scipy ndimage package
    offset_image = snd.shift(offset_image, initial_shift)
    return offset_image, ref_points_target, ref_points_offset


def register_detect_rotation(target_image: np.ndarray, offset_image: np.ndarray, ref_points_target, ref_points_offset):

    # num z_planes
    num_z = target_image.shape[0]
    # since input images are 3D volumes, we must take the mid z_plane
    target_image_mid = target_image[int(num_z/2)]
    offset_image_mid = offset_image[int(num_z/2)]

    # Define centre of circle we will be registering
    centre = np.mean(ref_points_target)
    # Radius is determined to be the min distance from both centroids to their respective boundaries
    centroid_target = np.mean(ref_points_target)
    centroid_offset = np.mean(ref_points_offset)
    # Dimensions are same for both images
    image_dims = np.array(target_image_mid.shape, dtype=int)
    # Now we choose the biggest circle fitting in full target, biggest circle fitting in offset, then take smaller one
    radius_target = np.min([centroid_target, image_dims - centroid_target])
    radius_offset = np.min([centroid_offset, image_dims - centroid_offset])
    radius = int(np.min([radius_target, radius_offset]))
    # Now crop the images around this centre with this min radius
    target_image_mid = target_image_mid[centre[0] - radius:centre[0] + radius, centre[1] - radius:centre[1] + radius]
    offset_image_mid = offset_image_mid[centre[0] - radius:centre[0] + radius, centre[1] - radius:centre[1] + radius]

    # Now apply the rotation detection algorithm on these. As these are 2 DAPIs, we shouldn't need
    # to do any processing on them. The only thing we should do is window them.
    # Apply Hann Window to Images 1 and 2
    target_image_mid = target_image_mid * (window('hann', target_image_mid.shape) ** 0.1)
    offset_image_mid = offset_image_mid * (window('hann', offset_image_mid.shape) ** 0.1)
    angle, error = base.detect_rotation(target_image_mid, offset_image_mid)
    angle_rad = angle * 2 * np.pi / 360

    # Now we apply this to this yx shift to the 3D volume. We'll only need about 20% of the volume around the middle to
    # do the next step so crop before we rotate
    target_image_cropped = target_image[int(0.4 * num_z):int(0.6 * num_z)]
    offset_image_cropped = offset_image[int(0.4 * num_z):int(0.6 * num_z)]
    # Update num_z
    num_z = target_image_cropped.shape[0]
    for z in range(num_z):
        offset_image_cropped[z] = rotate(offset_image_cropped[z], -angle_rad)

    return target_image_cropped, offset_image_cropped