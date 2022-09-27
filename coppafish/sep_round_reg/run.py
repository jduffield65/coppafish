from typing import List, Tuple, Optional
from coppafish.pipeline.run import initialize_nb, run_extract, run_find_spots, run_stitch
from coppafish.sep_round_reg import base
import numpy as np
from base import detect_rotation, populate
from skimage.transform import rotate
from skimage.filters import window
from skimage.registration import phase_cross_correlation
import tifffile as tf
import napari
from PIL import Image
from tqdm import tqdm

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp


def run_sep_round_reg(config_file: str, config_file_full: str):
    """
    Bridge function to run everything. Currently incomplete.
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
    detect_rigid_transform(dapi_full, dapi_sep)


def register_manual_shift(target_image: np.ndarray, offset_image: np.ndarray):
    """
    Function to manually shift images using reference points on image 1 and on image 2.
    Args:
        target_image: The volume that we are trying to register to. (np.ndarray)
        offset_image: The offset volume. (np.ndarray)

    Returns:
        target_image: unchanged target image (np.ndarray)
        ref_points_target: The reference points chosen on the target image mid z-plane by the user. (np.ndarray)
        ref_points_offset: The reference points chosen on the offset image mid z-plane by the user. (np.ndarray)
    """
    # Take mid z-planes
    target_image_mid = target_image[target_image.shape[0] // 2]
    offset_image_mid = offset_image[offset_image.shape[0] // 2]
    # Normalise middle z-planes
    target_image_mid = target_image_mid / np.max(target_image_mid)
    offset_image_mid = offset_image_mid / np.max(offset_image_mid)
    # open napari interface and allow user to choose points
    initial_shift, ref_points_target, ref_points_offset = base.manual_shift2(target_image_mid, offset_image_mid)
    ref_points_target = np.array(ref_points_target, dtype='int')
    ref_points_offset = np.array(ref_points_offset, dtype='int')
    # Convert initial shift into 3D
    initial_shift = np.insert(initial_shift, 0, 0)
    # Apply this shift using custom-built shift function
    offset_image = base.shift(offset_image, -initial_shift)
    return offset_image, ref_points_target, ref_points_offset, initial_shift


def register_detect_rotation(target_image: np.ndarray, offset_image: np.ndarray):
    """
    This function takes in 2 images and finds the yx rotation between them.
    Args:
        target_image: The volume that we are trying to register to. (np.ndarray)
        offset_image: The offset volume. (np.ndarray)

    Returns:
        angle: angle taking the target mid z-plane to the offset mid z-plane (float)
        target_image: the normalised target_image (np.ndarray)
        offset_image: The normalised offset image now with each z-plane rotated by -angle (np.ndarray)

    """
    # since input images are 3D volumes, and we're only looking for the yx rotation between them, we take the mid
    # z-planes
    target_image_mid = np.sum(target_image[target_image.shape[0]//2 - 1:target_image.shape[0]//2+1], axis=0)
    offset_image_mid = np.sum(offset_image[offset_image.shape[0]//2 - 1:offset_image.shape[0]//2+1], axis=0)

    # Now apply the rotation detection algorithm on these.
    # Apply Hann Window to Images 1 and 2
    target_image_mid = target_image_mid * (window('hann', target_image_mid.shape) ** 0.1)
    offset_image_mid = offset_image_mid * (window('hann', offset_image_mid.shape) ** 0.1)
    # use rotation detection algorithm defined in base
    angle = detect_rotation(target_image_mid, offset_image_mid)

    # rotate each z-plane by this amount
    for z in range(offset_image.shape[0]):
        offset_image[z] = rotate(offset_image[z], -angle)

    return angle, target_image, offset_image


def register_detect_rotation_loc(target_image: np.ndarray, offset_image: np.ndarray, ref_points_target: np.ndarray,
                                 ref_points_offset: np.ndarray):
    """
    This function takes in two volumes and a set of points within these volumes that should correspond to each other.
    It then looks locally around those points and finds the best rotation taking a small neighbourhood around each ref
    point in the target volume to a corresponding neighbourhood around ref points in the target volume
    Args:
        target_image: The volume that we are trying to register to. (np.ndarray)
        offset_image: The offset volume. (np.ndarray)
        ref_points_target: The reference points chosen on the target image mid z-plane by the user in the manual shift
        stage of the algorithm. (np.ndarray)
        ref_points_offset: The reference points chosen on the offset image mid z-plane by the user in the manual shift
        stage of the algorithm.(np.ndarray)

    Returns:
        angle: angle taking the target mid z-plane to the offset mid z-plane (float)
        target_image: the normalised target_image (np.ndarray)
        offset_image: The normalised offset image now with each z-plane rotated by -angle (np.ndarray)
        radius: the radius used around each ref point to complete this registration (int)

    """
    # since input images are 3D volumes, and we're only looking for the yx rotation between them, we take the mid
    # z-plane
    target_image_mid = target_image[target_image.shape[0] // 2]
    offset_image_mid = offset_image[offset_image.shape[0] // 2]

    # Dimensions are same for both images
    image_dims = np.array(target_image_mid.shape, dtype=int)
    # Now we choose the radius of the neighbourhoods we'll be registering
    radius = 250
    # create a vector to store all the angles found
    angle_vec = np.zeros(ref_points_target.shape[0])
    # Now we'll detect rotations at each of our ref_points
    for i in range(ref_points_target.shape[0]):
        # crop the images and apply Hann windows
        target_sample = target_image_mid[ref_points_target[i, 0] - radius: ref_points_target[i, 0] + radius,
                        ref_points_target[i, 1] - radius: ref_points_target[i, 1] + radius] * \
                        (window('hann', [2 * radius, 2 * radius]) ** 0.1)
        offset_sample = offset_image_mid[ref_points_offset[i, 0] - radius: ref_points_offset[i, 0] + radius,
                        ref_points_offset[i, 1] - radius: ref_points_offset[i, 1] + radius] * \
                        (window('hann', [2 * radius, 2 * radius]) ** 0.1)
        # use the rotation detection algorithm found in base to detect yx-rotation between these neighbourhoods
        angle_vec[i] = base.detect_rotation(target_sample, offset_sample)

    # Average across all these rotations
    angle = np.mean(angle_vec)

    # Now rotate each z-plane by this angle
    for z in range(offset_image.shape[0]):
        offset_image[z] = rotate(offset_image[z], -angle)

    # The previous step normalises offset_image so we must also normalise target_image
    target_image = target_image / np.max(target_image)

    return angle, target_image, offset_image, radius


def detect_rigid_transform(target_image: np.ndarray, offset_image: np.ndarray, ref_points_target=None,
                           ref_points_offset=None):
    """
    Semi-automated algorithm for finding a rigid transformation between two volumes. Steps are as follows.
    1.) Have the user manually select points on the mid z plane of both images to find the initial shift in yx
    2.) Detect the rotation between the first and second image mid z planes, apply this yx rotation to all z planes of
    the image
    3.) Use phase cross correlation to find optimal shift in 3D between both sets of ref points of images.

    Args:
        target_image: The volume that we are trying to register to. (np.ndarray)
        offset_image: The offset volume. (np.ndarray)
        ref_points_target: The reference points chosen on the target image mid z-plane by the user, if not given then
        the program will open a napari window and get the user to select them.
        ref_points_offset: The reference points chosen on the offset image mid z-plane by the user, if not given then
        the program will open a napari window and get the user to select them.

    Returns:
        initial_shift: The average shift taking ref_points target to ref_points offset (np.ndarray)
        angle: The angle taking the target image to the offset image (float)
        shift: The final small correction shift which takes the updated ref_points offset to their corresponding
        ref_points_target
    """

    # Start by cropping z- planes
    #  We'll only need about 40% of the volume around the middle for registration
    z_range = range(int(0.3 * target_image.shape[0]), int(0.7 * target_image.shape[0]))
    target_image = target_image[z_range]
    offset_image = offset_image[z_range]
    mid_z = len(z_range) // 2

    # Rescale so max = 1
    target_image = target_image / np.max(target_image)
    offset_image = offset_image / np.max(offset_image)

    # Step 1.) Manual Shift
    # If we have no reference points, then have the user select them
    if ref_points_offset is None or ref_points_target is None:
        offset_image, ref_points_target, ref_points_offset, initial_shift = register_manual_shift(target_image,
                                                                                                  offset_image)
    # if the reference points have been added then feed them in
    else:
        initial_shift = np.mean(ref_points_target - ref_points_offset, axis=0, dtype=int)
        # The initial shift should have 3 components, som
        initial_shift = np.insert(initial_shift, 0, 0)
        offset_image = base.shift(offset_image, initial_shift)

    # Step 2.)
    # Detect rotation between DAPI iamges. We take the features chosen in step 1 as our reference
    angle, target_image, offset_image, radius = \
        register_detect_rotation_loc(target_image, offset_image, ref_points_target, ref_points_offset)

    # We'll need the ref_points for the final stage.
    # We must make sure to apply the initial shift and rotation to the ref points on the offset image, updating their
    # position to the new coordinate system.
    ref_points_target = np.insert(ref_points_target, 0, mid_z, axis=1)
    # first, convert vectors to 2D and  translate the ref_points_offset by our initial shift
    ref_points_offset = ref_points_offset + [initial_shift[1:], initial_shift[1:], initial_shift[1:]]
    # next find the distance of these to the centre of the mid z_plane
    centre = np.array(offset_image[mid_z, :, :].shape) // 2
    ref_points_offset = ref_points_offset - centre
    # next rotate around this centre, as this is the point of rotation we used earlier on. First, build rotation matrix
    angle_rad = angle * np.pi / 180
    R = [[np.cos(angle_rad), np.sin(angle_rad)],
         [-np.sin(angle_rad), np.cos(angle_rad)]]
    ref_points_offset = np.matmul(ref_points_offset, np.transpose(R)).astype(int)
    # Now shift ref_points_offset back away from centre and add back the z_coord
    ref_points_offset = ref_points_offset + centre
    ref_points_offset = np.insert(ref_points_offset, 0, mid_z, axis=1)

    # Step 3: 3D shift detection. First create a matrix which will store 3 shifts as columns, these shifts will be the
    # best shift taking the updated (rotated + initially shifted) reference points on the offset image, to the target
    # image. This step generally produces small shifts that are just minor corrections to the large initial shift made
    # earlier.
    # First create a matrix for the shifts we will find with the phase cross correlation method, also store the errors
    # and phase differences associated with these shifts.
    pcc_shifts = np.zeros((3, ref_points_target.shape[0]))
    errors = np.zeros(3)
    phase_diffs = np.zeros(3)
    # Since we are detecting these shifts locally, we need to choose the size of the cubes containing the ref points
    # that we'd like to register. I've set these quite small for speed and because too much information can give
    # incorrect results
    z_range = 10
    y_range = 200
    x_range = 200
    # Next, apply the algorithm once for each of the ref_points
    for i in range(ref_points_target.shape[0]):
        small_target = target_image[ref_points_target[i, 0] - z_range: ref_points_target[i, 0] + z_range,
                       ref_points_target[i, 1] - y_range: ref_points_target[i, 1] + y_range,
                       ref_points_target[i, 2] - x_range: ref_points_target[i, 2] + x_range]
        small_offset = offset_image[ref_points_offset[i, 0] - z_range: ref_points_offset[i, 0] + z_range,
                       ref_points_offset[i, 1] - y_range: ref_points_offset[i, 1] + y_range,
                       ref_points_offset[i, 2] - x_range: ref_points_offset[i, 2] + x_range]
        pcc_shifts[i], errors[i], phase_diffs[i] = phase_cross_correlation(small_target, small_offset)

    # Next, take the mean of this shift and make it an integer as this is the only input our custom-built function
    # can take
    shift = np.mean(pcc_shifts, axis=0, dtype=int)

    # Next create a napari viewer and overlay image 1 and 2 so that we can review the quality of this registration
    # viewer = napari.Viewer()
    # Add image 1 and image 2 as image layers in Napari
    # viewer.add_image(target_image, blending='additive', colormap='bop orange')
    # viewer.add_image(offset_image, blending='additive', colormap='bop blue')
    # napari.run()

    return initial_shift, shift, angle


def apply_rigid_transform(image: np.ndarray, angle: float, initial_shift: np.ndarray, shift: np.ndarray):
    """

    Function to apply the rigid transform to the offset colume
    Args:
        image:  offset image to be rotated and translated (np.ndarray)
        angle: detected angle taking target to offset in yx (float)
        initial_shift: initial shift detected in register_detect_manual_shift (np.ndarray)
        shift: Final correction 3D shift (np.ndarray)

    Returns:
        image: The transformed image.
    """
    # Apply initial shift. SKImage is slow so use custom-built shift function found in base
    new_image = base.shift(image, initial_shift)

    # Apply yx rotation. SKImage is slow so use Pillow rotation which involves extraction to PIL, rotation and writing
    # back to ndarray
    for i in range(image.shape[0]):
        image2d = Image.fromarray(new_image[i])
        image2d = image2d.rotate(-angle)
        image2d = np.array(image2d)
        new_image[i] = image2d

    # Apply corrected shift
    new_image = base.shift(new_image, shift)

    return new_image


def register_retile(target_image: np.ndarray, offset_image: np.ndarray, rows: int, cols: int):
    """
    This function takes in the target image and an offset image which has already been registered using the rigid
    transform detection and application functions. These algorithms look for a global shift and rotation though and this
    doesn't capture all the local shifts and rotations that we observe. By retiling both images, we can deal with
    rotation in the z-direction too by shifting subtiles differently.
    Args:
        target_image: The volume that we are trying to register to. (np.ndarray)
        offset_image: The offset volume. (np.ndarray)
        rows: The number of rows we want in our retiling (int)
        cols: The number of cols we want in our retiling (int)
    Returns:
        registered_image: The offset volume after the application of subtile registration

    """
    # Store the phase cross correlation shifts found for each subtile in the array pcc_shifts. pcc_shifts[:,i,j] stores
    # the 3D vector for row i col j.
    pcc_shifts = np.zeros((3, rows, cols), dtype=int)
    # Find width of both volumes (as images have been expanded to have same dimensions)
    width = target_image.shape[2]
    # Find height of both volumes (as images have been expanded to have same dimensions)
    height = target_image.shape[1]
    # Create a zero array to be populated by the registered image.
    registered_image = np.zeros(target_image.shape)
    # Find mid z-plane (as an integer)
    mid_z = target_image.shape[0] // 2

    # Next we split up the work of detection and application of the shifts among the subtiles.

    # Create a progress bar to keep track of algorithms stage when running
    with tqdm(total=rows * cols) as pbar:
        pbar.set_description(f'Computing registration on subtiles')
        for i in range(rows):
            for j in range(cols):
                pbar.set_postfix({'subrow': i, 'subcol': j})
                # Extract the cropped subtiles. Downsample in x,y so that x,y,z have same pixel sizes. I'm not sure
                # if this changes the shifts found but intuitively it seems it would. In any case, it makes the code
                # run faster and is probably sufficient information

                new_iss_tile = target_image[mid_z - 5:mid_z + 5, i * height // rows: (i + 1) * height // rows: 3,
                               j * width // cols: (j + 1) * width // cols: 3]

                new_ifr_tile = offset_image[mid_z - 5:mid_z + 5, i * height // rows: (i + 1) * height // rows: 3,
                               j * width // cols: (j + 1) * width // cols: 3]

                # Detect the shift between subtile[i,j] on the offset image and on the target image using the pcc
                # algorithm
                pcc_shift = phase_cross_correlation(new_iss_tile, new_ifr_tile,
                                                    reference_mask=np.ones(new_iss_tile.shape, dtype=int) - np.isnan(
                                                        new_iss_tile),
                                                    moving_mask=np.ones(new_ifr_tile.shape, dtype=int) - np.isnan(
                                                        new_ifr_tile))
                # Next multiply the yx shift by the factor by which we downsampled. Don't do this to the z-shift though
                pcc_shifts[:, i, j] = np.array(3 * pcc_shift, dtype=int)
                pcc_shifts[0, i, j] = pcc_shifts[0, i, j] // 3
                # Update progress bar
                pbar.update(1)

    # Now apply this shift and populate the registered image array
    for i in range(rows):
        for j in range(cols):
            # Extract the cropped subtile
            new_ifr_tile = offset_image[:, i * height // rows: (i + 1) * height // rows,
                           j * width // cols: (j + 1) * width // cols]
            # Apply the shift. To eliminate a boundary of zeros around every subtile, we don't compute the shifted
            # image using the base.shift function but rather overlay the image onto the new registered_image array.
            # Watch out for double covering!

            # This isn't working at the moment so fix tomorrow!
            # starting_point = [0, i * height // rows, j * width // cols] + pcc_shifts[:, i, j]
            # registered_image = populate(new_ifr_tile, registered_image, starting_point)
            new_ifr_tile = base.shift(new_ifr_tile, pcc_shifts[:, i, j])
            registered_image[:, i * height // rows: (i + 1) * height // rows,
                           j * width // cols: (j + 1) * width // cols] = new_ifr_tile

    return registered_image

# Load in our data
dapi_partial_temp = np.load('C:/Users/Reilly/Desktop/dapi_partial_unreg.npy')
dapi_full_temp = np.load('C:/Users/Reilly/Desktop/dapi_full.npy')
# Now we need to ensure that the volumes have the same dimensions, take the biggest of each dimension across image 1
# and 2, create an array of zeros with these dimensions and populate the array as much as possible
big_container = np.maximum(dapi_full_temp.shape, dapi_partial_temp.shape)
dapi_partial = np.zeros(big_container)
dapi_full = np.zeros(big_container)
# now put these into the bigger arrays
dapi_partial[:dapi_partial_temp.shape[0], :dapi_partial_temp.shape[1], :dapi_partial_temp.shape[2]] = dapi_partial_temp
dapi_full[:dapi_full_temp.shape[0], :dapi_full_temp.shape[1], :dapi_full_temp.shape[2]] = dapi_full_temp
# Delete the temporary files to free up memory
del dapi_partial_temp, dapi_full_temp

# These are our saved reference points, if you don't have any, the code will select them manually
rp_target = np.load('C:/Users/Reilly/Desktop/ref_points_target.npy')
rp_offset = np.load('C:/Users/Reilly/Desktop/ref_points_offset.npy')

# Detect the initial shift, global rotation and final global correction shift
initial_shift, shift, angle = detect_rigid_transform(target_image=dapi_full, offset_image=dapi_partial,
                                                     ref_points_target=rp_target, ref_points_offset=rp_offset)

# Apply the rigid transformation we have detected
dapi_partial = apply_rigid_transform(image=dapi_partial, angle=angle, initial_shift=initial_shift, shift=shift)

# Now start the retiling procedure. This will allow us to bypass stitching irregularities and take care of rotation in
# z, by simply applying different vertical shifts at different subtiles!

# Keep ratio of rows to columns the same as in the original notebook

dapi_partial_new = register_retile(dapi_full, dapi_partial, 16, 24)

viewer = napari.Viewer()
# Add image 1 and image 2 as image layers in Napari
viewer.add_image(dapi_full, blending='additive', colormap='bop orange')
viewer.add_image(dapi_partial, blending='additive', colormap='bop blue')
viewer.add_image(dapi_partial_new, blending='additive')
napari.run()

# print(shift, angle, error)
# astro = rgb2gray(data.astronaut())
# astro_new = base.shift(astro, [10, 20])
# astro_new = rotate(astro_new, 7)
#
# base.manual_shift(astro, astro_new)
