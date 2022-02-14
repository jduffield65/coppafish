import os
from .. import setup, utils
from . import set_basic_info, extract_and_filter, find_spots, stitch, register_initial, register, reference_spots, \
    call_reference_spots
import warnings
from typing import Union


def run_pipeline(config_file: str) -> setup.Notebook:
    """
    Bridge function to run every step of the pipeline.

    Args:
        config_file: Path to config file.

    Returns:
        `Notebook` containing all information gathered during the pipeline.
    """
    nb = initialize_nb(config_file)
    config = setup.get_config(config_file)
    nb = run_extract(nb, config)
    nb = run_find_spots(nb, config)
    nb = run_stitch(nb, config)
    nb = run_register(nb, config)
    nb = run_reference_spots(nb, config)
    return nb


def initialize_nb(config_file: str) -> setup.Notebook:
    """
    Quick function which creates a `Notebook` and adds `file_names` and `basic_info` pages
    before saving.

    If `Notebook` already exists and contains these pages, it will just be returned.

    Args:
        config_file: Path to config file.

    Returns:
        `Notebook` containing `file_names` and `basic_info` pages.
    """
    config = setup.get_config(config_file)
    nb_path = os.path.join(config['file_names']['output_dir'], 'notebook.npz')
    nb = setup.Notebook(nb_path, config_file)
    if not all(nb.has_page(["file_names", "basic_info"])):
        nbp_file, nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        nb += nbp_file
        nb += nbp_basic
    else:
        warnings.warn('basic_info', utils.warnings.NotebookPageWarning)
        warnings.warn('file_names', utils.warnings.NotebookPageWarning)
    return nb


def run_extract(nb: setup.Notebook, config: Union[dict, str]) -> setup.Notebook:
    """
    This runs the `extract_and_filter` step of the pipeline to produce the tiff files in the tile directory.

    `extract` and `extract_debug` pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `file_names` and `basic_info` pages.
        config: Path to config file or Dictionary obtained from config file containing key
            `'extract'` which is another dict.

    Returns:
        `Notebook` with `extract` and `extract_debug` pages added.
    """
    if isinstance(config, str):
        # sometimes I guess it will be easier to pass the config file name so deal with that case here too
        config = setup.get_config(config)
    if not all(nb.has_page(["extract", "extract_debug"])):
        nbp, nbp_debug = extract_and_filter(config['extract'], nb.file_names, nb.basic_info)
        nb += nbp
        nb += nbp_debug
    else:
        warnings.warn('extract', utils.warnings.NotebookPageWarning)
        warnings.warn('extract_debug', utils.warnings.NotebookPageWarning)
    return nb


def run_find_spots(nb: setup.Notebook, config: Union[dict, str]) -> setup.Notebook:
    """
    This runs the `find_spots` step of the pipeline to produce point cloud from each tiff file in the tile directory.

    `find_spots` page added to the `Notebook` before saving.

    If `Notebook` already contains this page, it will just be returned.

    Args:
        nb: `Notebook` containing `extract` page.
        config: Path to config file or Dictionary obtained from config file containing key
            `'find_spots'` which is another dict.

    Returns:
        `Notebook` with `find_spots` page added.
    """
    if isinstance(config, str):
        config = setup.get_config(config)
    if not nb.has_page("find_spots"):
        nbp = find_spots(config['find_spots'], nb.file_names, nb.basic_info, nb.extract.auto_thresh)
        nb += nbp
    else:
        warnings.warn('find_spots', utils.warnings.NotebookPageWarning)
    return nb


def run_stitch(nb: setup.Notebook, config: Union[dict, str]) -> setup.Notebook:
    """
    This runs the `stitch` step of the pipeline to produce origin of each tile
    such that a global coordinate system can be built. Also saves stitched DAPI and reference channel images.

    `stitch_debug` page added to the `Notebook` before saving.

    If `Notebook` already contains this page, it will just be returned.
    If stitched images already exist, they won't be created again.

    Args:
        nb: `Notebook` containing `find_spots` page.
        config: Path to config file or Dictionary obtained from config file containing key
            `'stitch'` which is another dict.

    Returns:
        `Notebook` with `stitch_debug` page added.
    """
    if isinstance(config, str):
        config = setup.get_config(config)
    if not nb.has_page("stitch_debug"):
        nbp_debug = stitch(config['stitch'], nb.basic_info, nb.find_spots.spot_details)
        nb += nbp_debug
    else:
        warnings.warn('stitch_debug', utils.warnings.NotebookPageWarning)
    if nb.file_names.big_dapi_image is not None and not os.path.isfile(nb.file_names.big_dapi_image):
        # save stitched dapi
        utils.tiff.save_stitched(nb.file_names.big_dapi_image, nb.file_names, nb.basic_info,
                                 nb.stitch_debug.tile_origin, nb.basic_info.anchor_round,
                                 nb.basic_info.dapi_channel)
    if nb.file_names.big_anchor_image is not None and not os.path.isfile(nb.file_names.big_anchor_image):
        # save stitched reference round/channel
        utils.tiff.save_stitched(nb.file_names.big_anchor_image, nb.file_names, nb.basic_info,
                                 nb.stitch_debug.tile_origin, nb.basic_info.ref_round,
                                 nb.basic_info.ref_channel)
    return nb


def run_register(nb: setup.Notebook, config: Union[dict, str]) -> setup.Notebook:
    """
    This runs the `register_initial` step of the pipeline to find shift between ref round/channel to each imaging round
    for each tile. It then runs the `register` step of the pipeline which uses this as a starting point to get
    the affine transforms to go from the ref round/channel to each imaging round/channel for every tile.

    `register_initial_debug`, `register` and `register_debug` pages are added to the `Notebook` before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `extract` page.
        config: Path to config file or Dictionary obtained from config file containing keys
            `'register_initial'` and `'register'` which each are also dictionaries.

    Returns:
        `Notebook` with `register_initial_debug`,`register` and `register_debug` pages added.
    """
    if isinstance(config, str):
        config = setup.get_config(config)
    if not nb.has_page("register_initial_debug"):
        nbp_initial_debug = register_initial(config['register_initial'], nb.basic_info,
                                             nb.find_spots.spot_details)
        nb += nbp_initial_debug
    else:
        warnings.warn('register_initial_debug', utils.warnings.NotebookPageWarning)
    if not all(nb.has_page(["register", "register_debug"])):
        nbp, nbp_debug = register(config['register'], nb.basic_info, nb.find_spots.spot_details,
                                  nb.register_initial_debug.shift)
        nb += nbp
        nb += nbp_debug
    else:
        warnings.warn('register', utils.warnings.NotebookPageWarning)
        warnings.warn('register_debug', utils.warnings.NotebookPageWarning)
    return nb


def run_reference_spots(nb: setup.Notebook, config: Union[dict, str]) -> setup.Notebook:
    """
    This runs the `reference_spots` step of the pipeline to get the intensity of each spot on the reference
    round/channel in each imaging round/channel. The `call_spots` step of the pipeline is then run to produce the
    `bleed_matrix`, `bled_code` for each gene and the gene assignments of the spots on the reference round.

    `ref_spots` and `call_spots` pages are added to the Notebook before saving.

    If `Notebook` already contains these pages, it will just be returned.

    Args:
        nb: `Notebook` containing `stitch_debug` and `register` pages.
        config: Path to config file or Dictionary obtained from config file containing key
            `'call_spots'` which is another dict.

    Returns:
        `Notebook` with `ref_spots` and `call_spots` pages added.
    """
    if isinstance(config, str):
        config = setup.get_config(config)
    if not all(nb.has_page(["ref_spots", "call_spots"])):
        nbp_ref_spots = reference_spots(nb.file_names, nb.basic_info, nb.find_spots.spot_details,
                                        nb.stitch_debug.tile_origin, nb.register.transform)
        nbp, nbp_ref_spots = call_reference_spots(config['call_spots'], nb.file_names, nb.basic_info, nbp_ref_spots,
                                                  nb.extract.hist_values, nb.extract.hist_counts)
        nb += nbp_ref_spots
        nb += nbp
    else:
        warnings.warn('ref_spots', utils.warnings.NotebookPageWarning)
        warnings.warn('call_spots', utils.warnings.NotebookPageWarning)
    return nb


if __name__ == '__main__':
    ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/3d/anne_3d.ini'
    # ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/2d/anne_2d.ini'
    notebook = run_pipeline(ini_file)
