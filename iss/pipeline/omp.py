from .. import utils, setup
from .. import find_spots as fs
from tqdm import tqdm
import numpy as np
import numpy_indexed
from ..setup.notebook import NotebookPage
from ..extract import scale
from ..spot_colors import get_all_pixel_colors
from ..call_spots import get_spot_intensity
from .. import omp
from typing import Tuple
from sklearn.neighbors import NearestNeighbors
import os


def call_spots_omp(config: dict, config_call_spots: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage,
                   nbp_call_spots: NotebookPage, tile_origin: np.ndarray,
                   transform: np.ndarray) -> NotebookPage:
    nbp = setup.NotebookPage("omp")

    # use bled_codes with gene efficiency incorporated and only use_rounds/channels
    rc_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    bled_codes = np.moveaxis(np.moveaxis(nbp_call_spots.bled_codes_ge, 0, -1)[rc_ind], -1, 0)
    norm_bled_codes = np.linalg.norm(bled_codes, axis=(1, 2))
    if np.abs(norm_bled_codes-1).max() > 1e-6:
        raise ValueError("nbp_call_spots.bled_codes_ge don't all have an L2 norm of 1 over "
                         "use_rounds and use_channels.")
    n_genes, n_rounds_use, n_channels_use = bled_codes.shape
    dp_norm_shift = config_call_spots['dp_norm_shift'] * np.sqrt(n_rounds_use)

    if nbp_basic.is_3d:
        detect_radius_z = config['radius_z']
    else:
        detect_radius_z = None

    use_tiles = nbp_basic.use_tiles.copy()
    if not os.path.isfile(nbp_file.omp_spot_shape):
        # Set tile order so do central tile first because better to compute spot_shape from central tile.
        t_centre = scale.select_tile(nbp_basic.tilepos_yx, nbp_basic.use_tiles)
        t_centre_ind = np.where(np.array(nbp_basic.use_tiles)==t_centre)[0][0]
        use_tiles[0], use_tiles[t_centre_ind] = use_tiles[t_centre_ind], use_tiles[0]
        spot_shape = None
    else:
        nbp.shape_tile = None
        nbp.shape_spot_local_yxz = None
        nbp.shape_spot_gene_no = None
        nbp.spot_shape_float = None
        # -1 because saved as uint16 so convert 0, 1, 2 to -1, 0, 1.
        spot_shape = utils.tiff.load(nbp_file.omp_spot_shape).astype(int) - 1

    spot_info = np.zeros((0, 7), dtype=int)
    spot_coefs = np.zeros((0, n_genes))
    # While iterating through tiles, only save info for rounds/channels using - add all rounds/channels back in later
    spot_background_coefs = np.zeros((0, n_channels_use))
    spot_colors = np.zeros((0, n_rounds_use, n_channels_use), dtype=int)
    for t in use_tiles:
        # this returns colors in use_rounds/channels only and no nan.
        pixel_colors, pixel_yxz = get_all_pixel_colors(t, transform, nbp_file, nbp_basic)

        # Only keep pixels with significant absolute intensity to save memory.
        # absolute because important to find negative coefficients as well.
        pixel_intensity = get_spot_intensity(np.abs(pixel_colors / nbp_call_spots.color_norm_factor[rc_ind]))
        keep = pixel_intensity > config['initial_intensity_thresh']
        pixel_colors = pixel_colors[keep]
        pixel_yxz = pixel_yxz[keep]
        del pixel_intensity, keep

        pixel_coefs, pixel_background_coefs = \
            omp.get_all_coefs(pixel_colors / nbp_call_spots.color_norm_factor[rc_ind], bled_codes,
                              config_call_spots['background_weight_shift'],  dp_norm_shift, config['dp_thresh'],
                              config['alpha'], config['beta'],  config['max_genes'], config['weight_coef_fit'])
        if spot_shape is None:
            nbp.shape_tile = t
            spot_yxz, spot_gene_no = omp.get_spots(pixel_coefs, pixel_yxz, config['radius_xy'], detect_radius_z)
            z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
            spot_shape, spots_used, nbp.spot_shape_float = \
                omp.spot_neighbourhood(pixel_coefs, pixel_yxz, spot_yxz, spot_gene_no, config['shape_max_size'],
                                       config['shape_pos_neighbour_thresh'], config['shape_isolation_dist'], z_scale,
                                       config['shape_sign_thresh'])

            nbp.shape_spot_local_yxz = spot_yxz[spots_used]
            nbp.shape_spot_gene_no = spot_gene_no[spots_used]
            utils.tiff.save(spot_shape + 1, nbp_file.omp_spot_shape)  # add 1 so can be saved as uint16.
            del spot_yxz, spot_gene_no, spots_used

        spot_info_t = \
            omp.get_spots(pixel_coefs, pixel_yxz, config['radius_xy'], detect_radius_z, 0, spot_shape,
                          config['initial_pos_neighbour_thresh'])
        n_spots = spot_info_t[0].shape[0]
        spot_info_t = np.concatenate([spot_var.reshape(n_spots, -1) for spot_var in spot_info_t], axis=1)
        spot_info_t = np.append(spot_info_t, np.ones((n_spots, 1), dtype=int) * t, axis=1)

        # find index of each spot in pixel array to add colors and coefs
        pixel_index = numpy_indexed.indices(pixel_yxz, spot_info_t[:, :3])

        # append this tile info to all tile info
        spot_colors = np.append(spot_colors, pixel_colors[pixel_index], axis=0)
        del pixel_colors
        spot_coefs = np.append(spot_coefs, pixel_coefs[pixel_index], axis=0)
        del pixel_coefs
        spot_background_coefs = np.append(spot_background_coefs, pixel_background_coefs[pixel_index], axis=0)
        del pixel_background_coefs, pixel_index
        spot_info = np.append(spot_info, spot_info_t, axis=0)
        del spot_info_t

    nbp.spot_shape = spot_shape

    # find duplicate spots as those detected on a tile which is not tile centre they are closest to
    # Do this in 2d as overlap is only 2d
    tile_centres = tile_origin + nbp_basic.tile_centre
    tree_tiles = NearestNeighbors(n_neighbors=1).fit(tile_centres[:, :2])
    spot_global_yxz = spot_info[:, :3] + tile_origin[spot_info[:, 6]]
    _, all_nearest_tile = tree_tiles.kneighbors(spot_global_yxz[:, :2])
    not_duplicate = all_nearest_tile.flatten() == spot_info[:, 6]

    # Add spot info to notebook page
    nbp.local_yxz = spot_info[not_duplicate, :3]
    nbp.tile = spot_info[not_duplicate, 6]

    # spot_colors and background_coef have nans if use_rounds / use_channels used.
    n_spots = np.sum(not_duplicate)
    spot_colors_full = np.ones((nbp_basic.n_rounds, nbp_basic.n_channels, n_spots), dtype=float) * np.nan
    spot_colors_full[rc_ind] = np.moveaxis(spot_colors[not_duplicate], 0, -1)
    nbp.colors = np.moveaxis(spot_colors_full, -1, 0)
    background_coefs_full = np.ones((n_spots, nbp_basic.n_channels)) * np.nan
    background_coefs_full[np.ix_(np.arange(n_spots), nbp_basic.use_channels)] = spot_background_coefs[not_duplicate]
    nbp.background_coef = background_coefs_full

    nbp.coef = spot_coefs[not_duplicate]
    nbp.gene_no = spot_info[not_duplicate, 3]
    nbp.n_neighbours_pos = spot_info[not_duplicate, 4]
    nbp.n_neighbours_neg = spot_info[not_duplicate, 5]
    nbp.intensity = get_spot_intensity(spot_colors[not_duplicate] / nbp_call_spots.color_norm_factor[rc_ind])

    # Add quality thresholds to notebook page
    nbp.score_multiplier = config['score_multiplier']
    nbp.score_thresh = config['score_thresh']
    nbp.intensity_thresh = config['intensity_thresh']

    return nbp
