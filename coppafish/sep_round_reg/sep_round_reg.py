from typing import List, Tuple, Optional
from coppafish.register import get_single_affine_transform
from coppafish.pipeline.run import initialize_nb, run_extract, run_find_spots, run_stitch
from coppafish import setup, utils, Notebook
from coppafish.call_spots import get_non_duplicate
from coppafish.stitch import compute_shift
from coppafish.find_spots import get_isolated_points
from coppafish.spot_colors import apply_transform
from coppafish.plot.register.shift import view_shifts
from ..sep_round_reg import rotation_detection as rd
import numpy as np
import os
import warnings
import matplotlib

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
    # nb_full = Notebook(config_file_full)
    # Both elements in the sum below are 3 columns (y,x,z positions) by num_ref_spots rows, one for each spot found on
    # the anchor channel. The first term we are summing is just the local spot position, while the second is the shift
    # to be applied to the tile it's found on
    global_yxz_full = nb_full.ref_spots.local_yxz + nb_full.stitch.tile_origin[nb_full.ref_spots.tile]

    # run pipeline to get as far as a set of global coordinates for the separate round anchor.
    nb = initialize_nb(config_file)
    run_extract(nb)
    run_find_spots(nb)
    run_stitch(nb)
    config = nb.get_config()

    # if not nb.has_page("stitch"):
    #     nbp_stitch = stitch(config['stitch'], nb.basic_info, nb.find_spots.spot_details, nb.find_spots.spot_no)
    #     nb += nbp_stitch
    # else:
    #     warnings.warn('stitch', utils.warnings.NotebookPageWarning)

    # scale z coordinate so in units of xy pixels as other 2 coordinates are.
    z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
    # z_scale_full = nb_full.basic_info.pixel_size_z / nb_full.basic_info.pixel_size_xy
    # need both z_scales to be the same for final transform_image to work. Does not always seem to be the case though
    z_scale_full = z_scale

    # Compute centre of stitched image, as when running PCR, coordinates are centred first.
    yx_origin = np.round(nb.stitch.tile_origin[:, :2]).astype(int)
    z_origin = np.round(nb.stitch.tile_origin[:, 2]).astype(int).flatten()
    yx_size = np.max(yx_origin, axis=0) + nb.basic_info.tile_sz
    if nb.basic_info.is_3d:
        z_size = z_origin.max() + nb.basic_info.nz
        image_centre = np.floor(np.append(yx_size, z_size) / 2).astype(int)
    else:
        image_centre = np.append(np.floor(yx_size / 2).astype(int), 0)

    if not nb.has_page('reg_to_anchor_info'):
        nbp = setup.NotebookPage('reg_to_anchor_info')
        if transform is not None:
            nbp.transform = transform
        else:
            # remove duplicate spots

            # Start by reading in all spots
            spot_local_yxz = nb.find_spots.spot_details
            # Next we create an array which is of length n_spots which will tell us which tile we are on
            spot_tile = np.zeros(spot_local_yxz.shape[0]).astype(dtype=int)
            # This will contain num_spots per tile
            spots_per_tile = np.zeros(nb.basic_info.n_tiles).astype(dtype=int)
            # This will start with a 0 and contain the row index of spot_details where a new tile begins
            new_tile_indices = np.zeros(nb.basic_info.n_tiles + 1).astype(dtype=int)
            for i in range(nb.basic_info.n_tiles):
                spots_per_tile[i] = np.sum(nb.find_spots.spot_no[i, :, :])
                new_tile_indices[i + 1] = np.sum(spots_per_tile[0:i + 1])
            # Now loop through all spots and see which interval of new_tile_indices they lie between
            for i in range(spot_local_yxz.shape[0]):
                for j in range(nb.basic_info.n_tiles):
                    if new_tile_indices[j] <= i <= new_tile_indices[j + 1]:
                        spot_tile[i] = j
                        break

            # Some tiles will be double counted, get all those which are not
            not_duplicate = get_non_duplicate(nb.stitch.tile_origin, nb.basic_info.use_tiles,
                                              nb.basic_info.tile_centre, spot_local_yxz, spot_tile)
            global_yxz = spot_local_yxz[not_duplicate] + nb.stitch.tile_origin[spot_tile[not_duplicate]]

            # Only keep isolated points far from neighbour
            if nb.basic_info.is_3d:
                neighb_dist_thresh = config['register']['neighb_dist_thresh_3d']
            else:
                neighb_dist_thresh = config['register']['neighb_dist_thresh_2d']

            isolated = get_isolated_points(global_yxz * [1, 1, z_scale], 2 * neighb_dist_thresh)
            isolated_full = get_isolated_points(global_yxz_full * [1, 1, z_scale_full], 2 * neighb_dist_thresh)
            global_yxz = global_yxz[isolated, :]
            global_yxz_full = global_yxz_full[isolated_full, :]

            # Get the stitched anchor directories for both the full and the partial notebook
            # stitched_anchor_full_dir = '//128.40.224.65/SoyonHong/Christina Maat/ISS Data + Analysis/E-2207-001_full/' \
            #                            'Cp/output/anchor_image.npz'
            # stitched_anchor_dir = '//128.40.224.65/Shared Projects/ISS/MG/Reilly/Dataset 1/anchor_image.npz'
            stitched_anchor_full_dir = nb_full.file_names.big_anchor_image
            stitched_anchor_dir = nb.file_names.big_anchor_image

            # Get the stitched anchor image for the full notebook
            mid_z_plane = int(len(nb_full.basic_info.use_z) / 2)
            stitched_anchor_full_read = np.load(stitched_anchor_full_dir)
            stitched_anchor_full = stitched_anchor_full_read.f.arr_0
            stitched_anchor_full = stitched_anchor_full[mid_z_plane-5:mid_z_plane+5, :, :]

            # Get the stitched anchor image for the partial notebook
            mid_z_plane = int(len(nb.basic_info.use_z) / 2)
            stitched_anchor_read = np.load(stitched_anchor_dir)
            stitched_anchor = stitched_anchor_read.f.arr_0
            stitched_anchor = stitched_anchor[mid_z_plane-5:mid_z_plane+5, :, :]

            # Find the side length of the maximal square which fits in both images
            square_length = min(stitched_anchor_full.shape[0], stitched_anchor_full.shape[1], stitched_anchor.shape[0],
                                stitched_anchor.shape[0])

            # Process the images to make them easier to analyse
            stitched_anchor_full = rd.process_image(stitched_anchor_full,
                                                    z_planes=np.arange(stitched_anchor_full.shape[0]), gamma=4, y=0,
                                                    x=0, length=square_length)
            stitched_anchor = rd.process_image(stitched_anchor,
                                                    z_planes=np.arange(stitched_anchor.shape[0]), gamma=4, y=0,
                                                    x=0, length=square_length)
            # Return the angle and the error associated with detection which we must rotate the partial anchor by to
            # get the full anchor (angle is anticlockwise and in degrees)
            angle, error = rd.detect_rotation(stitched_anchor, stitched_anchor_full)
            # Convert angle to radians
            angle_rad = angle * 2 * np.pi / 360
            # Create rotation matrix to apply to the anchor spots in the partial notebook
            rotation_matrix = np.zeros((3, 3))
            rotation_matrix[0:2, 0:2] = \
                np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
            # Next, shift origin to centre of image, apply rotation and then transform back
            for i in range(global_yxz.shape[0]):
                global_yxz[i] = global_yxz[i] - image_centre
                global_yxz[i] = np.matmul(rotation_matrix, global_yxz[i])
                global_yxz[i] = global_yxz[i] + image_centre

            # get initial shift from separate round to the full image
            nbp.shift, nbp.shift_score, nbp.shift_score_thresh, debug_info = \
                get_shift(config['register_initial'], global_yxz, global_yxz_full,
                          z_scale, z_scale_full, nb.basic_info.is_3d)

            # view_shifts(debug_info['shifts_2d'], debug_info['scores_2d'], debug_info['shifts_3d'],
            #             debug_info['scores_3d'], nbp.shift, nbp.shift_score_thresh)

            # Get affine transform from separate round to full image
            start_transform = np.eye(4, 3)  # no scaling just shift to start off icp
            start_transform[3] = nbp.shift * [1, 1, z_scale]
            nbp.transform, nbp.n_matches, nbp.error, nbp.is_converged = \
                get_single_affine_transform(global_yxz, global_yxz_full, z_scale, z_scale_full,
                                            start_transform, neighb_dist_thresh, image_centre)
        nb += nbp  # save results of transform found
    else:
        nbp = nb.reg_to_anchor_info
        if transform is not None:
            if (transform != nb.reg_to_anchor_info.transform).any():
                raise ValueError(f"transform given is:\n{transform}.\nThis differs "
                                 f"from nb.reg_to_anchor_info.transform:\n{nb.reg_to_anchor_info.transform}")

    # save all the images
    for c in channels_to_save:
        im_file = os.path.join(nb.file_names.output_dir, f'sep_round_channel{c}_transformed.npz')
        if c == nb.basic_info.ref_channel:
            from_nd2 = False
        else:
            from_nd2 = True
        image_stitch = utils.npy.save_stitched(None, nb.file_names, nb.basic_info, nb.stitch.tile_origin,
                                               nb.basic_info.ref_round, c, from_nd2,
                                               config['stitch']['save_image_zero_thresh'])

        image_transform = transform_image(image_stitch, nbp.transform, image_centre[:image_stitch.ndim], z_scale)
        if nb.basic_info.is_3d:
            # Put z axis first for saving
            image_transform = np.moveaxis(image_transform, -1, 0)
        np.savez_compressed(im_file, image_transform)


def get_shift(config: dict, spot_yxz_base: np.ndarray, spot_yxz_transform: np.ndarray, z_scale_base: float,
              z_scale_transform: float, is_3d: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Find shift from base to transform.

    Args:
        config: register_initial section of config file corresponding to spot_yxz_base.
        spot_yxz_base: Point cloud want to find the shift from.
            spot_yxz_base[:, 2] is the z coordinate in units of z-pixels.
        spot_yxz_transform: Point cloud want to find the shift to.
            spot_yxz_transform[:, 2] is the z coordinate in units of z-pixels.
        z_scale_base: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        z_scale_transform: Scaling to put z coordinates in same units as yx coordinates for spot_yxz_base.
        is_3d: Whether pipeline is 3D or not.

    Returns:
        `shift` - `float [shift_y, shift_x, shift_z]`.
            Best shift found.
        `shift_score` - `float`.
            Score of best shift found.
        `min_score` - `float`.
            Threshold score that was calculated, i.e. range of shifts searched changed until score exceeded this.
    """

    coords = ['y', 'x', 'z']
    shifts = {}
    for i in range(len(coords)):
        shifts[coords[i]] = np.arange(config['shift_min'][i],
                                      config['shift_max'][i] +
                                      config['shift_step'][i] / 2, config['shift_step'][i]).astype(int)
    if not is_3d:
        config['shift_widen'][2] = 0  # so don't look for shifts in z direction
        config['shift_max_range'][2] = 0
        shifts['z'] = np.array([0], dtype=int)
    shift, shift_score, shift_score_thresh, debug_info = \
        compute_shift(spot_yxz_base, spot_yxz_transform,
                      config['shift_score_thresh'], config['shift_score_thresh_multiplier'],
                      config['shift_score_thresh_min_dist'], config['shift_score_thresh_max_dist'],
                      config['neighb_dist_thresh'], shifts['y'], shifts['x'], shifts['z'],
                      config['shift_widen'], config['shift_max_range'], [z_scale_base, z_scale_transform],
                      config['nz_collapse'], config['shift_step'][2])
    return shift, np.asarray(shift_score), np.asarray(shift_score_thresh), debug_info


def transform_image(image: np.ndarray, transform: np.ndarray, image_centre: np.ndarray, z_scale: int) -> np.ndarray:
    """
    This transforms `image` to a new coordinate system by applying `transform` to every pixel in `image`.

    Args:
        image: `int [n_y x n_x (x n_z)]`.
            image which is to be transformed.
        transform: `float [4 x 3]`.
            Affine transform which transforms image which is applied to every pixel in image to form a
            new transformed image.
        image_centre: `int [image.ndim]`.
            Pixel coordinates were centred by subtracting this first when computing affine transform.
            So when applying affine transform, pixels will also be shifted by this amount.
            z centre i.e. `image_centre[2]` is in units of z-pixels.
        z_scale: Scaling to put z coordinates in same units as yx coordinates.

    Returns:
        `int [n_y x n_x (x n_z)]`.
            `image` transformed according to `transform`.

    """
    im_transformed = np.zeros_like(image)
    yxz = jnp.asarray(np.where(image != 0)).T.reshape(-1, image.ndim)
    image_values = image[tuple([yxz[:, i] for i in range(image.ndim)])]
    tile_size = jnp.asarray(im_transformed.shape)
    if image.ndim == 2:
        tile_size = jnp.append(tile_size, 1)
        image_centre = np.append(image_centre, 0)
        yxz = np.hstack((yxz, np.zeros((yxz.shape[0], 1))))

    yxz_transform, in_range = apply_transform(yxz, jnp.asarray(transform), jnp.asarray(image_centre), z_scale,
                                              tile_size)
    yxz_transform = np.asarray(yxz_transform[in_range])
    image_values = image_values[np.asarray(in_range)]
    im_transformed[tuple([yxz_transform[:, i] for i in range(image.ndim)])] = image_values
    return im_transformed

# if __name__ == "__main__":
#     channels = [0, 18]
#
#     run_sep_round_reg(config_file='sep_round_settings_example.ini',
#                       config_file_full='experiment_settings_example.ini',
#                       channels_to_save=channels)
