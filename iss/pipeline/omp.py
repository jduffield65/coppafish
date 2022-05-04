from .. import utils, setup
from .. import find_spots as fs
from tqdm import tqdm
import numpy as np
from ..setup.notebook import NotebookPage
from ..extract import scale
from ..spot_colors import get_all_pixel_colors
from ..call_spots import get_spot_intensity
from .. import omp
from typing import Tuple
import os


def call_spots_omp(config: dict, config_call_spots: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage,
                   nbp_call_spots: NotebookPage, tile_origin: np.ndarray,
                   transform: np.ndarray) -> NotebookPage:
    nbp = setup.NotebookPage("omp")

    # use bled_codes with gene efficiency incorporated.
    rc_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    bled_codes = np.moveaxis(np.moveaxis(nbp_call_spots.bled_codes_ge, 0, -1)[rc_ind], -1, 0)
    n_genes, n_rounds_use = bled_codes.shape[:2]
    dp_norm_shift = config_call_spots['dp_norm_shift'] * np.sqrt(n_rounds_use)

    if nbp_basic.is_3d:
        detect_radius_z = config['radius_z']
    else:
        detect_radius_z = None

    use_tiles = nbp_basic.use_tiles.copy()
    if not os.path.isfile(nbp_file.omp_spot_shape):
        # Set tile order so do central tile first because better to compute spot_shape from central tile.
        t_centre = scale.select_tile(nbp_basic.tilepos_yx, nbp_basic.use_tiles)
        use_tiles[0], use_tiles[t_centre] = use_tiles[t_centre], use_tiles[0]
        spot_shape = None
    else:
        nbp.shape_tile = None
        nbp.shape_spot_yxz = None
        nbp.shape_spot_gene_no = None
        nbp.spot_shape_float = None
        # -1 because saved as uint16 so convert 0, 1, 2 to -1, 0, 1.
        spot_shape = utils.tiff.load(nbp_file.omp_spot_shape).astype(int) - 1


    for t in use_tiles:
        pixel_colors, pixel_yxz = get_all_pixel_colors(t, transform, nbp_file, nbp_basic)
        pixel_colors = pixel_colors / nbp_call_spots.color_norm_factor[rc_ind]

        # Only keep pixels with significant absolute intensity to save memory.
        # absolute because important to find negative coefficients as well.
        pixel_intensity = get_spot_intensity(np.abs(pixel_colors))
        keep = pixel_intensity > config['initial_intensity_thresh']
        pixel_colors = pixel_colors[keep]
        pixel_yxz = pixel_yxz[keep]
        del pixel_intensity, keep

        pixel_coefs, pixel_background_coefs = \
            omp.get_all_coefs(pixel_colors, bled_codes, config_call_spots['background_weight_shift'],  dp_norm_shift,
                              config['dp_thresh'], config['alpha'], config['beta'],  config['max_genes'],
                              config['weight_coef_fit'])
        if spot_shape is None:
            nbp.shape_tile = t
            spot_yxzg = omp.detect_spots_all_genes(pixel_coefs, pixel_yxz, config['radius_xy'], detect_radius_z)
            z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
            spot_shape, spots_used, nbp.spot_shape_float = \
                omp.spot_neighbourhood(pixel_coefs, pixel_yxz, spot_yxzg, config['shape_max_size'],
                                       config['shape_pos_neighbour_thresh'], config['shape_isolation_dist'], z_scale,
                                       config['shape_sign_thresh'])
            # TODO: maybe for all coordinates, only save local coordinates and tile, then add tile_origin in situ to
            #  get global. That way, save int coordinates not float.
            nbp.shape_spot_yxz = spot_yxzg[spots_used, :3] + tile_origin[t]
            nbp.shape_spot_gene_no = spot_yxzg[spots_used, 3].astype(int)
            utils.tiff.save(spot_shape + 1, nbp_file.omp_spot_shape)  # add 1 so can be saved as uint16.



    nbp.spot_shape = spot_shape


    return nbp
