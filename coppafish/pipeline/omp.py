from .. import utils
import numpy as np
import numpy_indexed
from ..setup.notebook import NotebookPage
from ..extract import scale
from ..spot_colors import get_spot_colors, all_pixel_yxz
from ..call_spots import get_spot_intensity, get_non_duplicate
from .. import omp
from ..utils.base import round_any

import os
import warnings
from scipy import sparse
from typing import Optional
try:
    import jax.numpy as jnp
except ImportError:
    warnings.warn('Jax is not installed so call_spots_omp will be slow')
    import numpy as jnp


def call_spots_omp(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage,
                   nbp_call_spots: NotebookPage, tile_origin: np.ndarray,
                   transform: np.ndarray, shape_tile: Optional[int]) -> NotebookPage:
    """
    This runs orthogonal matching pursuit (omp) on every pixel to determine a coefficient for each gene at each pixel.

    From these gene coefficient images, a local maxima search is performed to find the position of spots for each gene.
    Various properties of the spots are then saved to determine the likelihood that the gene assignment is legitimate.

    See `'omp'` section of `notebook_comments.json` file for description of the variables in the omp page.

    Args:
        config: Dictionary obtained from `'omp'` section of config file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        nbp_call_spots:
        tile_origin: `float [n_tiles x 3]`.
            `tile_origin[t,:]` is the bottom left yxz coordinate of tile `t`.
            yx coordinates in `yx_pixels` and z coordinate in `z_pixels`.
            This is saved in the `stitch` notebook page i.e. `nb.stitch.tile_origin`.
        transform: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.
            `transform[t, r, c]` is the affine transform to get from tile `t`, `ref_round`, `ref_channel` to
            tile `t`, round `r`, channel `c`.
            This is saved in the register notebook page i.e. `nb.register.transform`.
        shape_tile: Tile to use to compute the expected shape of a spot in the gene coefficient images.
            Should be the tile, for which the most spots where found in the `call_reference_spots` step.
            If `None`, will be set to the centre tile.

    Returns:
        `NotebookPage[omp]` - Page contains gene assignments and info for spots using omp.

    """
    nbp = NotebookPage("omp")

    # use bled_codes with gene efficiency incorporated and only use_rounds/channels
    rc_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    bled_codes = np.moveaxis(np.moveaxis(nbp_call_spots.bled_codes_ge, 0, -1)[rc_ind], -1, 0)
    utils.errors.check_color_nan(bled_codes, nbp_basic)
    norm_bled_codes = np.linalg.norm(bled_codes, axis=(1, 2))
    if np.abs(norm_bled_codes - 1).max() > 1e-6:
        raise ValueError("nbp_call_spots.bled_codes_ge don't all have an L2 norm of 1 over "
                         "use_rounds and use_channels.")
    bled_codes = jnp.asarray(bled_codes)
    transform = jnp.asarray(transform)
    color_norm_factor = jnp.asarray(nbp_call_spots.color_norm_factor[rc_ind])
    n_genes, n_rounds_use, n_channels_use = bled_codes.shape
    dp_norm_shift = nbp_call_spots.dp_norm_shift * np.sqrt(n_rounds_use)

    if nbp_basic.is_3d:
        detect_radius_z = config['radius_z']
        n_z = nbp_basic.nz
    else:
        detect_radius_z = None
        n_z = 1
        config['use_z'] = np.arange(n_z)

    if config['use_z'] is not None:
        use_z_oob = [val for val in config['use_z'] if val < 0 or val >= n_z]
        if len(use_z_oob) > 0:
            raise utils.errors.OutOfBoundsError("use_z", use_z_oob[0], 0, n_z - 1)
        if len(config['use_z']) == 2:
            # use consecutive values if only 2 given.
            config['use_z'] = list(np.arange(config['use_z'][0], config['use_z'][1] + 1))
        use_z = np.array(config['use_z'])
    else:
        use_z = np.arange(n_z)

    # determine initial_intensity_thresh from average intensity over all pixels on central z-plane.
    nbp.initial_intensity_thresh = omp.get_initial_intensity_thresh(config, nbp_call_spots)

    use_tiles = np.array(nbp_basic.use_tiles.copy())
    if not os.path.isfile(nbp_file.omp_spot_shape):
        # Set tile order so do shape_tile first to compute spot_shape from it.
        if shape_tile is None:
            shape_tile = scale.central_tile(nbp_basic.tilepos_yx, nbp_basic.use_tiles)
        if shape_tile not in nbp_basic.use_tiles:
            raise ValueError(f"shape_tile, {shape_tile} is not in nbp_basic.use_tiles, {nbp_basic.use_tiles}")
        shape_tile_ind = np.where(np.array(nbp_basic.use_tiles) == shape_tile)[0][0]
        use_tiles[0], use_tiles[shape_tile_ind] = use_tiles[shape_tile_ind], use_tiles[0]
        spot_shape = None
    else:
        nbp.shape_tile = None
        nbp.shape_spot_local_yxz = None
        nbp.shape_spot_gene_no = None
        nbp.spot_shape_float = None
        # -1 because saved as uint16 so convert 0, 1, 2 to -1, 0, 1.
        spot_shape = np.load(nbp_file.omp_spot_shape)  # Put z to last index
        if spot_shape.ndim == 3:
            spot_shape = np.moveaxis(spot_shape, 0, 2)  # Put z to last index

    # Deal with case where algorithm has been run for some tiles and data saved
    if os.path.isfile(nbp_file.omp_spot_info) and os.path.isfile(nbp_file.omp_spot_coef):
        if spot_shape is None:
            raise ValueError(f'OMP information already exists for some tiles but spot_shape tiff file does not:\n'
                             f'{nbp_file.omp_spot_shape}\nEither add spot_shape tiff or delete the files:\n'
                             f'{nbp_file.omp_spot_info} and {nbp_file.omp_spot_coef}.')
        spot_coefs = sparse.load_npz(nbp_file.omp_spot_coef)
        spot_info = np.load(nbp_file.omp_spot_info)
        if spot_coefs.shape[0] > spot_info.shape[0]:
            # Case where bugged out after saving spot_coefs but before saving spot_info, delete all excess spot_coefs.
            warnings.warn(f"Have spot_coefs for {spot_coefs.shape[0]} spots but only spot_info for {spot_info.shape[0]}"
                          f" spots.\nSo deleting the excess spot_coefs and re-saving to {nbp_file.omp_spot_coef}.")
            spot_coefs = spot_coefs[:spot_info.shape[0]]
            sparse.save_npz(nbp_file.omp_spot_coef, spot_coefs)
        elif spot_coefs.shape[0] < spot_info.shape[0]:
            # If more spots in info than coefs then likely because duplicates removed from coefs but not spot_info.
            not_duplicate = get_non_duplicate(tile_origin, nbp_basic.use_tiles, nbp_basic.tile_centre,
                                              spot_info[:, :3], spot_info[:, 6])
            if not_duplicate.size == spot_info.shape[0]:
                warnings.warn(f'There were less spots in\n{nbp_file.omp_spot_info}\nthan\n{nbp_file.omp_spot_coef} '
                              f'because duplicates were deleted for spot_coefs but not for spot_info.\n'
                              f'Now, spot_info duplicates have also been deleted.')
                spot_info = spot_info[not_duplicate]
                np.save(nbp_file.omp_spot_info, spot_info)
            else:
                raise ValueError(f"Have spot_info for {spot_info.shape[0]} spots but only spot_coefs for "
                                 f"{spot_coefs.shape[0]}\nNeed to delete both {nbp_file.omp_spot_coef} and "
                                 f"{nbp_file.omp_spot_info} to get past this error.")
        else:
            prev_found_tiles = np.unique(spot_info[:, -1])
            use_tiles = np.setdiff1d(use_tiles, prev_found_tiles)
            warnings.warn(f'Already have OMP results for tiles {prev_found_tiles} so now just running on tiles '
                          f'{use_tiles}.')
        del spot_coefs, spot_info
    elif os.path.isfile(nbp_file.omp_spot_coef):
        # If only have information only file but not the other, need to delete all files and start again.
        raise ValueError(f'The file {nbp_file.omp_spot_coef} exists but the file {nbp_file.omp_spot_info} does not.\n'
                         f'Delete or re-name the file {nbp_file.omp_spot_coef} to run omp part from scratch.')
    elif os.path.isfile(nbp_file.omp_spot_info):
        raise ValueError(f'The file {nbp_file.omp_spot_info} exists but the file {nbp_file.omp_spot_coef} does not.\n'
                         f'Delete or re-name the file {nbp_file.omp_spot_info} to run omp part from scratch.')

    print(f'Finding OMP coefficients for all pixels on tiles {use_tiles}:')
    initial_pos_neighbour_thresh = config['initial_pos_neighbour_thresh']
    for t in use_tiles:
        pixel_yxz_t = np.zeros((0, 3), dtype=np.int16)
        pixel_coefs_t = sparse.csr_matrix(np.zeros((0, n_genes), dtype=np.float32))
        for z in use_z:
            print(f"Tile {np.where(use_tiles == t)[0][0] + 1}/{len(use_tiles)},"
                  f" Z-plane {np.where(use_z == z)[0][0] + 1}/{len(use_z)}")
            # While iterating through tiles, only save info for rounds/channels using
            # - add all rounds/channels back in later. This returns colors in use_rounds/channels only and no invalid.
            pixel_colors_tz, pixel_yxz_tz = get_spot_colors(all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz,
                                                                          int(z)), int(t), transform,
                                                            nbp_file, nbp_basic, return_in_bounds=True)
            if pixel_colors_tz.shape[0] == 0:
                continue
            pixel_colors_tz = pixel_colors_tz / color_norm_factor

            # Only keep pixels with significant absolute intensity to save memory.
            # absolute because important to find negative coefficients as well.
            # pixel_intensity_tz = get_spot_intensity(jnp.abs(pixel_colors_tz))
            pixel_intensity_tz = get_spot_intensity(jnp.abs(pixel_colors_tz))
            keep = pixel_intensity_tz > nbp.initial_intensity_thresh
            if not keep.any():
                continue
            pixel_colors_tz = pixel_colors_tz[keep]
            pixel_yxz_tz = pixel_yxz_tz[keep]
            del pixel_intensity_tz, keep

            pixel_coefs_tz = sparse.csr_matrix(
                omp.get_all_coefs(pixel_colors_tz, bled_codes,
                                  nbp_call_spots.background_weight_shift, dp_norm_shift, config['dp_thresh'],
                                  config['alpha'], config['beta'], config['max_genes'], config['weight_coef_fit'])[0])
            del pixel_colors_tz
            # Only keep pixels for which at least one gene has non-zero coefficient.
            keep = (np.abs(pixel_coefs_tz).max(axis=1) > 0).nonzero()[0]  # nonzero as is sparse matrix.
            if len(keep) == 0:
                continue
            # TODO: check order of np.asarray and keep, which is quicker - think this is quickest though
            pixel_yxz_t = np.append(pixel_yxz_t, np.asarray(pixel_yxz_tz[keep]), axis=0)
            del pixel_yxz_tz
            pixel_coefs_t = sparse.vstack((pixel_coefs_t, pixel_coefs_tz[keep]))
            del pixel_coefs_tz, keep

        if spot_shape is None:
            nbp.shape_tile = int(t)
            spot_yxz, spot_gene_no = omp.get_spots(pixel_coefs_t, pixel_yxz_t, config['radius_xy'], detect_radius_z)
            z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
            spot_shape, spots_used, spot_shape_float = \
                omp.spot_neighbourhood(pixel_coefs_t, pixel_yxz_t, spot_yxz, spot_gene_no, config['shape_max_size'],
                                       config['shape_pos_neighbour_thresh'], config['shape_isolation_dist'], z_scale,
                                       config['shape_sign_thresh'])
            if not np.isin(-1, spot_shape):
                # Rase error if computed average spot shape has no negative values
                error_file = nbp_file.omp_spot_shape.replace('.npy', '_float_ERROR.npy')
                if spot_shape_float.ndim == 3:
                    # put z axis to front before saving if 3D
                    np.save(error_file, np.moveaxis(spot_shape_float, 2, 0))
                else:
                    np.save(error_file, spot_shape_float)
                shape_neg_values = spot_shape_float[spot_shape_float < 0]
                message = f"Error when computing nb.omp.spot_shape:\n" \
                          f"Average spot_shape in OMP Coefficient images was found with {spots_used.size} spots.\n" \
                          f"However, it contains no pixels with a value of -1.\n" \
                          f"nb.omp.spot_shape_float was saved as\n{error_file}\n"
                if len(shape_neg_values) == 0:
                    message += "This contains no negative values either, so OMP section needs re-running with " \
                               "config['file_names']['omp_spot_shape'] specified"
                else:
                    max_neg_value = round_any(np.abs(shape_neg_values).max(), 0.001, 'floor')
                    message += f"OMP section needs re-running with\n" \
                               f"config['omp']['shape_sign_thresh'] < {max_neg_value} or " \
                               f"config['file_names']['omp_spot_shape'] specified"
                raise ValueError(message)
            nbp.spot_shape_float = spot_shape_float
            nbp.shape_spot_local_yxz = spot_yxz[spots_used]
            nbp.shape_spot_gene_no = spot_gene_no[spots_used]
            if spot_shape.ndim == 3:
                # put z axis to front before saving if 3D
                np.save(nbp_file.omp_spot_shape, np.moveaxis(spot_shape, 2, 0))
            else:
                np.save(nbp_file.omp_spot_shape, spot_shape)
            # already found spots so don't find again.
            spot_yxzg = np.append(spot_yxz, spot_gene_no.reshape(-1, 1), axis=1)
            del spot_yxz, spot_gene_no, spots_used
        else:
            spot_yxzg = None

        if initial_pos_neighbour_thresh is None:
            # Only save spots which have 10% of max possible number of positive neighbours
            initial_pos_neighbour_thresh = config['initial_pos_neighbour_thresh_param'] * np.sum(spot_shape > 0)
            initial_pos_neighbour_thresh = np.floor(initial_pos_neighbour_thresh)
            initial_pos_neighbour_thresh = int(np.clip(initial_pos_neighbour_thresh,
                                                       config['initial_pos_neighbour_thresh_min'],
                                                       config['initial_pos_neighbour_thresh_max']))
        spot_info_t = \
            omp.get_spots(pixel_coefs_t, pixel_yxz_t, config['radius_xy'], detect_radius_z, 0, spot_shape,
                          initial_pos_neighbour_thresh, spot_yxzg)
        del spot_yxzg
        n_spots = spot_info_t[0].shape[0]
        spot_info_t = np.concatenate([spot_var.reshape(n_spots, -1).astype(np.int16) for spot_var in spot_info_t],
                                     axis=1)
        spot_info_t = np.append(spot_info_t, np.ones((n_spots, 1), dtype=np.int16) * t, axis=1)

        # find index of each spot in pixel array to add colors and coefs
        pixel_index = numpy_indexed.indices(pixel_yxz_t, spot_info_t[:, :3])

        # append this tile info to all tile info
        if os.path.isfile(nbp_file.omp_spot_info) and os.path.isfile(nbp_file.omp_spot_coef):
            # After ran on one tile, need to load in spot_coefs and spot_info, append and then save again.
            spot_coefs = sparse.load_npz(nbp_file.omp_spot_coef)
            spot_coefs = sparse.vstack((spot_coefs, pixel_coefs_t[pixel_index]))
            del pixel_coefs_t, pixel_index
            sparse.save_npz(nbp_file.omp_spot_coef, spot_coefs)
            del spot_coefs
            spot_info = np.load(nbp_file.omp_spot_info)
            spot_info = np.append(spot_info, spot_info_t, axis=0)
            del spot_info_t
            np.save(nbp_file.omp_spot_info, spot_info)
            del spot_info
        else:
            # 1st tile, need to create files to save to
            sparse.save_npz(nbp_file.omp_spot_coef, pixel_coefs_t[pixel_index])
            del pixel_coefs_t, pixel_index
            np.save(nbp_file.omp_spot_info, spot_info_t.astype(np.int16))
            del spot_info_t

    nbp.spot_shape = spot_shape
    nbp.initial_pos_neighbour_thresh = initial_pos_neighbour_thresh

    spot_info = np.load(nbp_file.omp_spot_info)
    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    not_duplicate = get_non_duplicate(tile_origin, nbp_basic.use_tiles, nbp_basic.tile_centre,
                                      spot_info[:, :3], spot_info[:, 6])

    # Add spot info to notebook page
    nbp.local_yxz = spot_info[not_duplicate, :3]
    nbp.tile = spot_info[not_duplicate, 6]

    # Get colors, background_coef and intensity of final spots.
    n_spots = np.sum(not_duplicate)
    invalid_value = -nbp_basic.tile_pixel_value_shift
    # Only read in used colors first for background/intensity calculation.
    nd_spot_colors_use = np.ones((n_spots, n_rounds_use, n_channels_use), dtype=np.int32) * invalid_value
    for t in nbp_basic.use_tiles:
        in_tile = nbp.tile == t
        if np.sum(in_tile) > 0:
            nd_spot_colors_use[in_tile] = get_spot_colors(jnp.asarray(nbp.local_yxz[in_tile]), t,
                                                          transform, nbp_file, nbp_basic)

    spot_colors_norm = jnp.array(nd_spot_colors_use) / color_norm_factor
    nbp.intensity = np.asarray(get_spot_intensity(spot_colors_norm))
    del spot_colors_norm

    # When saving to notebook, include unused rounds/channels.
    nd_spot_colors = np.ones((n_spots, nbp_basic.n_rounds, nbp_basic.n_channels), dtype=np.int32) * invalid_value
    nd_spot_colors[np.ix_(np.arange(n_spots), nbp_basic.use_rounds, nbp_basic.use_channels)] = nd_spot_colors_use
    nbp.colors = nd_spot_colors
    del nd_spot_colors_use

    nbp.gene_no = spot_info[not_duplicate, 3]
    nbp.n_neighbours_pos = spot_info[not_duplicate, 4]
    nbp.n_neighbours_neg = spot_info[not_duplicate, 5]

    return nbp
