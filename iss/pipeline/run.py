import os
from .. import setup
from . import set_basic_info, extract_and_filter, find_spots, run_stitch, run_register_initial
from ..utils.tiff import save_stitched


def run_pipeline(config_file):
    config = setup.get_config(config_file)
    nb_path = os.path.join(config['file_names']['output_dir'], 'notebook.npz')
    nb = setup.Notebook(nb_path, config_file)
    if not min(nb.has_page(["file_names", "basic_info"])):
        nbp_file, nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        nb.add_page(nbp_file, do_nothing_if_exists=True)  # TODO: get rid of this flag
        nb.add_page(nbp_basic, True)
    if not min(nb.has_page(["extract", "extract_debug"])):
        nbp_extract, nbp_debug = extract_and_filter(
            config['extract'], nb['file_names'], nb['basic_info'])
        nb.add_page(nbp_extract, True)  # TODO get rid of params notebook page, just add all params that can be set by auto to debug.
        nb.add_page(nbp_debug, True)
    if not nb.has_page("find_spots"):
        nbp_find_spots = find_spots(config['find_spots'], nb['file_names'],
                                    nb['basic_info'], nb['extract']['auto_thresh'])
        nb.add_page(nbp_find_spots, True)
    if not nb.has_page("stitch_debug"):
        nbp_debug = run_stitch(config['stitch'], nb['basic_info'], nb['find_spots']['spot_details'])
        nb.add_page(nbp_debug, True)
    if nb['file_names']['big_dapi_image'] is not None and not os.path.isfile(nb['file_names']['big_dapi_image']):
        # save stitched dapi
        save_stitched(nb['file_names']['big_dapi_image'], nb['file_names'], nb['basic_info'],
                      nb['stitch_debug']['tile_origin'], nb['basic_info']['anchor_round'],
                      nb['basic_info']['dapi_channel'])
    if nb['file_names']['big_anchor_image'] is not None and not os.path.isfile(nb['file_names']['big_anchor_image']):
        # save stitched reference round/channel
        save_stitched(nb['file_names']['big_anchor_image'], nb['file_names'], nb['basic_info'],
                      nb['stitch_debug']['tile_origin'], nb['basic_info']['ref_round'],
                      nb['basic_info']['ref_channel'])
    if not nb.has_page("register_initial_debug"):
        nbp_debug = run_register_initial(config['register_initial'], nb['basic_info'], nb['find_spots']['spot_details'])
        nb.add_page(nbp_debug, True)
    return nb


if __name__ == '__main__':
    ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/3d/anne_3d.ini'
    # ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/2d/anne_2d.ini'
    notebook = run_pipeline(ini_file)
