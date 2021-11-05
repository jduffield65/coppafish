import os
from iss.setup.config import get_config
from iss.setup.notebook import Notebook, NotebookPage
from basic_info import set_basic_info
from extract_run import extract_and_filter


def run_pipeline(config_file):
    config = get_config(config_file)
    nb_path = os.path.join(config['file_names']['output_dir'], 'notebook.npz')
    nb = Notebook(nb_path, config_file)
    if not nb.has_page(["file_names", "basic_info"], contains_all=True):
        nbp_file, nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        nb.add_page(nbp_file, do_nothing_if_exists=True)
        nb.add_page(nbp_basic, True)
    if not nb.has_page(["extract", "extract_params", "extract_debug"], True):
        nbp_extract, nbp_params, nbp_debug = extract_and_filter(
            config['extract'], nb['file_names'], nb['basic_info'])
        nb.add_page(nbp_extract, True)
        nb.add_page(nbp_params, True)
        nb.add_page(nbp_debug, True)
    return nb


if __name__ == '__main__':
    # ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/3d/anne_3d.ini'
    ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/2d/anne_2d.ini'
    notebook = run_pipeline(ini_file)
