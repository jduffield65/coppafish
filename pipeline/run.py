from setup.config import get_config
from basic_info import set_basic_info


def run_pipeline(config_file):
    config = get_config(config_file)
    log = set_basic_info(config['file_names'], config['basic_info'])
    return log


if __name__ == '__main__':
    ini_file = '/Users/joshduffield/Documents/UCL/ISS/Python/play/anne_3d.ini'
    log = run_pipeline(ini_file)
