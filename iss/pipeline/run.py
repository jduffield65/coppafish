import os
from .. import setup, utils
from . import set_basic_info, extract_and_filter, find_spots, stitch, register_initial, register, get_reference_spots, \
    call_reference_spots, call_spots_omp
from ..find_spots import check_n_spots
from ..stitch import check_shifts_stitch, check_shifts_register
from ..call_spots import get_non_duplicate
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


def run_pipeline(config_file: str) -> setup.Notebook:
    """
    Bridge function to run every step of the pipeline.

    Args:
        config_file: Path to config file.

    Returns:
        `Notebook` containing all information gathered during the pipeline.
    """
    nb = initialize_nb(config_file)
    # spot_no = 371046
    run_extract(nb)
    run_find_spots(nb)
    run_stitch(nb)
    run_register(nb)
    run_reference_spots(nb)
    run_omp(nb)
    return nb


def initialize_nb(config_file: str) -> setup.Notebook:
    """
    Quick function which creates a `Notebook` and adds `basic_info` page before saving.
    `file_names` page will be added automatically as soon as `basic_info` page is added.
    If `Notebook` already exists and contains these pages, it will just be returned.

    Args:
        config_file: Path to config file.

    Returns:
        `Notebook` containing `file_names` and `basic_info` pages.
    """
    nb = setup.Notebook(config_file=config_file)
    config = nb.get_config()
    if not nb.has_page("basic_info"):
        nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        nb += nbp_basic
    else:
        warnings.warn('basic_info', utils.warnings.NotebookPageWarning)
    return nb


def run_extract(nb: setup.Notebook):
    """
    This runs the `extract_and_filter` step of the pipeline to produce the tiff files in the tile directory.

    `extract` and `extract_debug` pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `file_names` and `basic_info` pages.

    """
    if not all(nb.has_page(["extract", "extract_debug"])):
        config = nb.get_config()
        nbp, nbp_debug = extract_and_filter(config['extract'], nb.file_names, nb.basic_info)
        nb += nbp
        nb += nbp_debug
    else:
        warnings.warn('extract', utils.warnings.NotebookPageWarning)
        warnings.warn('extract_debug', utils.warnings.NotebookPageWarning)


def run_find_spots(nb: setup.Notebook):
    """
    This runs the `find_spots` step of the pipeline to produce point cloud from each tiff file in the tile directory.

    `find_spots` page added to the `Notebook` before saving.

    If `Notebook` already contains this page, it will just be returned.

    Args:
        nb: `Notebook` containing `extract` page.

    """
    if not nb.has_page("find_spots"):
        config = nb.get_config()
        nbp = find_spots(config['find_spots'], nb.file_names, nb.basic_info, nb.extract.auto_thresh)
        nb += nbp
        check_n_spots(nb)  # error if too few spots
    else:
        warnings.warn('find_spots', utils.warnings.NotebookPageWarning)


def run_stitch(nb: setup.Notebook):
    """
    This runs the `stitch` step of the pipeline to produce origin of each tile
    such that a global coordinate system can be built. Also saves stitched DAPI and reference channel images.

    `stitch` page added to the `Notebook` before saving.

    If `Notebook` already contains this page, it will just be returned.
    If stitched images already exist, they won't be created again.

    Args:
        nb: `Notebook` containing `find_spots` page.

    """
    config = nb.get_config()
    if not nb.has_page("stitch"):
        nbp_debug = stitch(config['stitch'], nb.basic_info, nb.find_spots.spot_details)
        nb += nbp_debug
        check_shifts_stitch(nb)  # error if too many bad shifts between tiles
    else:
        warnings.warn('stitch', utils.warnings.NotebookPageWarning)
    if nb.file_names.big_dapi_image is not None and not os.path.isfile(nb.file_names.big_dapi_image):
        # save stitched dapi
        # Will load in from nd2 file if nb.extract_debug.r_dapi is None i.e. if no DAPI filtering performed.
        utils.npy.save_stitched(nb.file_names.big_dapi_image, nb.file_names, nb.basic_info,
                                nb.stitch.tile_origin, nb.basic_info.anchor_round,
                                nb.basic_info.dapi_channel, nb.extract_debug.r_dapi is None,
                                config['stitch']['save_image_zero_thresh'])
    if nb.file_names.big_anchor_image is not None and not os.path.isfile(nb.file_names.big_anchor_image):
        # save stitched reference round/channel
        utils.npy.save_stitched(nb.file_names.big_anchor_image, nb.file_names, nb.basic_info,
                                nb.stitch.tile_origin, nb.basic_info.ref_round,
                                nb.basic_info.ref_channel, False, config['stitch']['save_image_zero_thresh'])


def run_register(nb: setup.Notebook):
    """
    This runs the `register_initial` step of the pipeline to find shift between ref round/channel to each imaging round
    for each tile. It then runs the `register` step of the pipeline which uses this as a starting point to get
    the affine transforms to go from the ref round/channel to each imaging round/channel for every tile.

    `register_initial`, `register` and `register_debug` pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `extract` page.

    """
    config = nb.get_config()
    if nb.has_page("register_initial_debug"):
        # deal with old notebook files where page was called "register_initial_debug" instead of
        # "register_initial". This will trigger a save after name change too.
        nb.change_page_name("register_initial_debug", "register_initial")
    if not nb.has_page("register_initial"):
        nbp_initial = register_initial(config['register_initial'], nb.basic_info,
                                             nb.find_spots.spot_details)
        nb += nbp_initial
        check_shifts_register(nb)  # error if too many bad shifts between rounds
    else:
        warnings.warn('register_initial', utils.warnings.NotebookPageWarning)
    if not all(nb.has_page(["register", "register_debug"])):
        nbp, nbp_debug = register(config['register'], nb.basic_info, nb.find_spots.spot_details,
                                  nb.register_initial.shift)
        nb += nbp
        nb += nbp_debug
    else:
        warnings.warn('register', utils.warnings.NotebookPageWarning)
        warnings.warn('register_debug', utils.warnings.NotebookPageWarning)


def run_reference_spots(nb: setup.Notebook):
    """
    This runs the `reference_spots` step of the pipeline to get the intensity of each spot on the reference
    round/channel in each imaging round/channel. The `call_spots` step of the pipeline is then run to produce the
    `bleed_matrix`, `bled_code` for each gene and the gene assignments of the spots on the reference round.

    `ref_spots` and `call_spots` pages are added to the Notebook before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `stitch` and `register` pages.

    """
    if not nb.has_page('ref_spots'):
        nbp = get_reference_spots(nb.file_names, nb.basic_info, nb.find_spots.spot_details,
                                  nb.stitch.tile_origin, nb.register.transform)
        nb += nbp  # save to Notebook with gene_no, score, score_diff, intensity = None.
                   # These will be added in call_reference_spots
    else:
        warnings.warn('ref_spots', utils.warnings.NotebookPageWarning)
    if not nb.has_page("call_spots"):
        if not os.path.isfile(nb.file_names.code_book):
            # Raise error here if code_book file does not exist as will be needed in call_reference_spots
            raise ValueError(f"The code_book file:\n{nb.file_names.code_book}\ndoes not exist. "
                             f"Change it in the config file and re-run.")
        config = nb.get_config()
        # nb.ref_spots.finalized = False  # so can set gene_no, score, score_diff, intensity.
        nbp, nbp_ref_spots = call_reference_spots(config['call_spots'], nb.file_names, nb.basic_info, nb.ref_spots,
                                                  nb.extract.hist_values, nb.extract.hist_counts, nb.register.transform)
        nb += nbp
        # only raise error after saving to notebook if spot_colors have nan in wrong places.
        utils.errors.check_color_nan(nb.ref_spots.colors, nb.basic_info)
    else:
        warnings.warn('call_spots', utils.warnings.NotebookPageWarning)


def run_omp(nb: setup.Notebook):
    """
    This runs the orthogonal matching pursuit section of the pipeline as an alternate method to determine location of
    spots and their gene identity.
    It achieves this by fitting multiple gene bled codes to each pixel to find a coefficient for every gene at
    every pixel. Spots are then local maxima in these gene coefficient images.

    `omp` page is added to the Notebook before saving.

    Args:
        nb: `Notebook` containing `call_spots` page.

    """
    if not nb.has_page("omp"):
        config = nb.get_config()
        nbp = call_spots_omp(config['omp'], nb.file_names, nb.basic_info, nb.call_spots,
                             nb.stitch.tile_origin, nb.register.transform)
        nb += nbp

        # Update omp_info files after omp notebook page saved into notebook
        # Save only non-duplicates - important spot_coefs saved first for exception at start of call_spots_omp
        # which can deal with case where duplicates removed from spot_coefs but not spot_info.
        # After re-saving here, spot_coefs[s] should be the coefficients for gene at nb.omp.local_yxz[s]
        # i.e. indices should match up.
        spot_info = np.load(nb.file_names.omp_spot_info)
        not_duplicate = get_non_duplicate(nb.stitch.tile_origin, nb.basic_info.use_tiles, nb.basic_info.tile_centre,
                                          spot_info[:, :3], spot_info[:, 6])
        spot_coefs = sparse.load_npz(nb.file_names.omp_spot_coef)
        sparse.save_npz(nb.file_names.omp_spot_coef, spot_coefs[not_duplicate])
        np.save(nb.file_names.omp_spot_info, spot_info[not_duplicate])

        # only raise error after saving to notebook if spot_colors have nan in wrong places.
        utils.errors.check_color_nan(nbp.colors, nb.basic_info)
    else:
        warnings.warn('omp', utils.warnings.NotebookPageWarning)
