import utils.nd2
from extract.base import *
from extract.filter import get_pixel_length, hanning_diff, disk_strel
from extract.scale import get_scale
import numpy as np


def extract_and_filter(config, log_file, log_basic):
    """

    :param config:
    :param log_file:
    :param log_basic:
    :return:
    """
    '''initialise log object'''
    # initialise log object with info from config
    log_extract = config
    # initialise output of this part of pipeline as 'vars' key
    log_extract['vars'] = {'auto_thresh': np.zeros((log_basic['n_tiles'], log_basic['n_channels'],
                                                    log_basic['n_rounds'] + log_basic['n_extra_rounds'])),
                           'hist_values': np.arange(-log_basic['tile_pixel_value_shift'], np.iinfo(np.uint16).max -
                                                    log_basic['tile_pixel_value_shift'] + 2, 1)}
    log_extract['vars']['hist_counts'] = np.zeros((len(log_extract['vars']['hist_values']), log_basic['n_channels'],
                                                   log_basic['n_rounds'] + log_basic['n_extra_rounds']), dtype=int)
    # initialise debugging info as 'diagnostic' key
    log_extract['diagnostic'] = {
        'n_clip_pixels': np.zeros_like(log_extract['vars']['auto_thresh'], dtype=int),
        'clip_extract_scale': np.zeros_like(log_extract['vars']['auto_thresh'])
    }

    '''update config params in log object'''
    if config['r1'] is None:
        log_extract['r1'] = get_pixel_length(log_extract['r1_auto_microns'], log_basic['pixel_size']['xy'])
    if config['r2'] is None:
        log_extract['r2'] = log_extract['r1'] * 2
    if config['r_dapi'] is None:
        log_extract['r_dapi'] = get_pixel_length(log_extract['r_dapi_auto_microns'], log_basic['pixel_size']['xy'])

    filter = hanning_diff(log_extract['r1'], log_extract['r2'])
    filter_dapi = disk_strel(config['r_dapi'])

    if config['scale'] is None:
        get_scale()

    '''get rounds to iterate over'''
    if log_basic['anchor_round'] is not None:
        # always have anchor as first round after imaging rounds
        round_files = log_file['round'] + [log_file['anchor']]
        use_rounds = log_basic['use_rounds'] + [log_basic['n_rounds']]
    else:
        round_files = log_file['round']
        use_rounds = log_basic['use_rounds']

    for r in use_rounds:
        im_file = os.path.join(log_file['input_dir'], round_files[r] + log_file['raw_extension'])
        wait_for_data(im_file, config['wait_time'])

    return log_extract
