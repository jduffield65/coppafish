import os
from .. import setup
from . import set_basic_info, extract_and_filter, find_spots, run_stitch


def run_pipeline(config_file):
    config = setup.get_config(config_file)
    nb_path = os.path.join(config['file_names']['output_dir'], 'notebook.npz')
    nb = setup.Notebook(nb_path, config_file)
    if not min(nb.has_page(["file_names", "basic_info"])):
        nbp_file, nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        nb.add_page(nbp_file, do_nothing_if_exists=True)
        nb.add_page(nbp_basic, True)
    if not min(nb.has_page(["extract", "extract_params", "extract_debug"])):
        nbp_extract, nbp_params, nbp_debug = extract_and_filter(
            config['extract'], nb['file_names'], nb['basic_info'])
        nb.add_page(nbp_extract, True)
        nb.add_page(nbp_params, True)
        nb.add_page(nbp_debug, True)
    if not min(nb.has_page(["find_spots", "find_spots_params"])):
        nbp_find_spots, nbp_params = find_spots(config['find_spots'], nb['file_names'],
                                                nb['basic_info'], nb['extract']['auto_thresh'])
        nb.add_page(nbp_find_spots, True)
        nb.add_page(nbp_params, True)
    if not min(nb.has_page(["stitch", "stitch_params", "stitch_debug"])):
        nbp_stitch, nbp_params, nbp_debug = run_stitch(config['stitch'], nb['file_names'], nb['basic_info'],
                                                       nb['find_spots']['spot_details'])
    return nb


if __name__ == '__main__':
    # ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/3d/anne_3d.ini'
    ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/2d/anne_2d.ini'
    notebook = run_pipeline(ini_file)
