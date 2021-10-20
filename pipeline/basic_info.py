import utils.nd2
import numpy as np
import os
import utils.errors
from setup.tile_details import get_tilepos, get_tile_file_names


def set_basic_info(config_file, config_basic):
    """
    Adds info from 'file_name' and 'basic_info' sections of config file
    to log object.
    To file_name, the following is added:
    tile
    To basic_info, the following is added:
    n_rounds, n_extra_rounds, n_tiles, n_channels, nz, tile_sz, tilepos_yx, pixel_size
    Also use_rounds, use_tiles, use_channels and use_z are changed if they were None.

    :param config_file: 'file_name' key of config dictionary
    :param config_basic: 'basic_info' key of config dictionary.
    :return:
        log: dictionary with 'file_name and 'basic_info'
    """
    # initialise log object with info from config
    log = {'file_names': config_file, 'basic_info': config_basic}

    # get round info from config file
    n_rounds = len(config_file['round'])
    n_extra_rounds = int(config_basic['anchor_channel'] is not None)
    log['basic_info']['use_rounds'] = config_basic['use_rounds']
    if log['basic_info']['use_rounds'] is None:
        log['basic_info']['use_rounds'] = list(np.arange(n_rounds))
    utils.errors.out_of_bounds('config-basic_info-use_rounds', log['basic_info']['use_rounds'], 0, n_rounds-1)

    # load in metadata of nd2 file corresponding to first round
    first_round_raw = os.path.join(config_file['input_dir'], config_file['round'][0]+config_file['raw_extension'])
    images = utils.nd2.load(first_round_raw)
    images = utils.nd2.update_metadata(images)
    # get tile info
    tile_sz = images.sizes['x']
    n_tiles = images.sizes['v']
    log['basic_info']['use_tiles'] = config_basic['use_tiles']
    if log['basic_info']['use_tiles'] is None:
        log['basic_info']['use_tiles'] = list(np.arange(n_tiles))
    utils.errors.out_of_bounds('config-basic_info-use_tiles', log['basic_info']['use_tiles'], 0, n_tiles - 1)
    # get channel info
    n_channels = images.sizes['c']
    log['basic_info']['use_channels'] = config_basic['use_channels']
    if log['basic_info']['use_channels'] is None:
        log['basic_info']['use_channels'] = list(np.arange(n_channels))
    utils.errors.out_of_bounds('config-basic_info-use_channels', log['basic_info']['use_channels'], 0, n_channels - 1)
    # get z info
    log['basic_info']['use_z'] = config_basic['use_z']
    if log['basic_info']['use_z'] is None:
        log['basic_info']['use_z'] = list(np.arange(images.sizes['z']))
    if config_basic['ignore_first_z_plane'] and 0 in log['basic_info']['use_z']:
        log['basic_info']['use_z'].remove(0)
    nz = len(log['basic_info']['use_z'])  # number of z planes in tiff file (not necessarily the same as in nd2)
    utils.errors.out_of_bounds('config-basic_info-use_z', log['basic_info']['use_z'], 0, images.sizes['z'] - 1)

    tilepos_yx = get_tilepos(images.metadata['xy_pos'], tile_sz)
    if config_file['anchor'] is not None:
        # always have anchor as first round after imaging rounds
        round_files = config_file['round'] + [config_file['anchor']]
    else:
        round_files = config_file['round']
    if config_basic['3d']:
        tile_names = get_tile_file_names(config_file['tile_dir'], round_files, tilepos_yx['nd2'], n_channels)
    else:
        tile_names = get_tile_file_names(config_file['tile_dir'], round_files, tilepos_yx['nd2'])

    log['file_names']['tile'] = tile_names  # tiff tile file paths numpy array [n_tiles x n_rounds (x n_channels if 3D)]
    log['basic_info']['n_rounds'] = n_rounds  # int, number of imaging rounds
    log['basic_info']['n_extra_rounds'] = n_extra_rounds  # int, number of anchor and other rounds
    log['basic_info']['tile_sz'] = tile_sz  # xy dimension of tiles in pixels.
    log['basic_info']['n_tiles'] = n_tiles  # int, number of tiles
    log['basic_info']['n_channels'] = n_channels  # int, number of imaging channels
    log['basic_info']['nz'] = nz  # number of z-planes used to make tiff images
    log['basic_info']['tilepos_yx'] = tilepos_yx  # dictionary, yx coordinate of tile with ['tiff'] and ['nd2'] index.
    # dictionary , pixel size in microns in xy and z directions.
    log['basic_info']['pixel_size'] = {'xy': images.metadata['pixel_microns'], 'z': images.metadata['pixel_microns_z']}
    return log
