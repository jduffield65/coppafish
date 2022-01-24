import os
from .. import setup, utils
from . import set_basic_info, extract_and_filter, find_spots, stitch, register_initial, register, reference_spots
import warnings


def run_pipeline(config_file):
    nb = initialize_nb(config_file)
    config = setup.get_config(config_file)
    nb = run_extract(nb, config)
    nb = run_find_spots(nb, config)
    nb = run_stitch(nb, config)
    nb = run_register(nb, config)
    nb = run_reference_spots(nb)
    return nb


def initialize_nb(config_file):
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


def run_extract(nb, config):
    if isinstance(config, str):
        # sometimes I guess it will be easier to pass the config file name so deal with that case here too
        config = setup.get_config(config)
    if not all(nb.has_page(["extract", "extract_debug"])):
        nbp, nbp_debug = extract_and_filter(config['extract'], nb['file_names'], nb['basic_info'])
        nb += nbp
        nb += nbp_debug
    else:
        warnings.warn('extract', utils.warnings.NotebookPageWarning)
        warnings.warn('extract_debug', utils.warnings.NotebookPageWarning)
    return nb


def run_find_spots(nb, config):
    if isinstance(config, str):
        config = setup.get_config(config)
    if not nb.has_page("find_spots"):
        nbp = find_spots(config['find_spots'], nb['file_names'], nb['basic_info'], nb['extract']['auto_thresh'])
        nb += nbp
    else:
        warnings.warn('find_spots', utils.warnings.NotebookPageWarning)
    return nb


def run_stitch(nb, config):
    if isinstance(config, str):
        config = setup.get_config(config)
    if not nb.has_page("stitch_debug"):
        nbp_debug = stitch(config['stitch'], nb['basic_info'], nb['find_spots']['spot_details'])
        nb += nbp_debug
    else:
        warnings.warn('stitch_debug', utils.warnings.NotebookPageWarning)
    if nb['file_names']['big_dapi_image'] is not None and not os.path.isfile(nb['file_names']['big_dapi_image']):
        # save stitched dapi
        utils.tiff.save_stitched(nb['file_names']['big_dapi_image'], nb['file_names'], nb['basic_info'],
                                 nb['stitch_debug']['tile_origin'], nb['basic_info']['anchor_round'],
                                 nb['basic_info']['dapi_channel'])
    if nb['file_names']['big_anchor_image'] is not None and not os.path.isfile(nb['file_names']['big_anchor_image']):
        # save stitched reference round/channel
        utils.tiff.save_stitched(nb['file_names']['big_anchor_image'], nb['file_names'], nb['basic_info'],
                                 nb['stitch_debug']['tile_origin'], nb['basic_info']['ref_round'],
                                 nb['basic_info']['ref_channel'])
    return nb


def run_register(nb, config):
    if isinstance(config, str):
        config = setup.get_config(config)
    if not nb.has_page("register_initial_debug"):
        nbp_initial_debug = register_initial(config['register_initial'], nb['basic_info'],
                                             nb['find_spots']['spot_details'])
        nb += nbp_initial_debug
    else:
        warnings.warn('register_initial_debug', utils.warnings.NotebookPageWarning)
    if not all(nb.has_page(["register", "register_debug"])):
        nbp, nbp_debug = register(config['register'], nb['basic_info'], nb['find_spots']['spot_details'],
                                  nb['register_initial_debug']['shift'])
        nb += nbp
        nb += nbp_debug
    else:
        warnings.warn('register', utils.warnings.NotebookPageWarning)
        warnings.warn('register_debug', utils.warnings.NotebookPageWarning)
    return nb


def run_reference_spots(nb):
    if not nb.has_page("ref_spots"):
        nbp = reference_spots(nb['file_names'], nb['basic_info'], nb['find_spots']['spot_details'],
                              nb['stitch_debug']['tile_origin'], nb['register']['transform'])
        nb += nbp
    else:
        warnings.warn('ref_spots', utils.warnings.NotebookPageWarning)
    return nb


if __name__ == '__main__':
    ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/3d/anne_3d.ini'
    # ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/2d/anne_2d.ini'
    notebook = run_pipeline(ini_file)
