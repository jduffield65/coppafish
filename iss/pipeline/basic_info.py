import numpy as np
import os
import iss.utils.errors
from ..utils import errors
import iss.utils.nd2
from iss.setup.tile_details import get_tilepos, get_tile_file_names
from iss.setup.notebook import NotebookPage
import warnings


def set_basic_info(config_file, config_basic):
    """
    Adds info from 'file_name' and 'basic_info' sections of config file
    to log object.
    To file_name, the following is added:
    tile
    To basic_info, the following is added:
    anchor_round, n_rounds, n_extra_rounds, n_tiles, n_channels, nz, tile_sz, tilepos_yx, pixel_size
    Also use_rounds, use_tiles, use_channels and use_z are changed if they were None.

    :param config_file: 'file_name' key of config dictionary
    :param config_basic: 'basic_info' key of config dictionary.
    :return:
        log: dictionary with 'file_name and 'basic_info'
    """
    # remove file extension from round and anchor file names if it is present
    config_file['round'] = [r.replace(config_file['raw_extension'], '') for r in config_file['round']]
    if config_file['anchor'] is not None:
        config_file['anchor'] = config_file['anchor'].replace(config_file['raw_extension'], '')

    # initialise log object with info from config
    nbp_file = NotebookPage("file_names", config_file)
    nbp_basic = NotebookPage("basic_info", config_basic)

    # get round info from config file
    n_rounds = len(config_file['round'])
    if config_basic['use_rounds'] is None:
        nbp_basic['use_rounds'] = list(np.arange(n_rounds))
    nbp_basic['use_rounds'].sort()  # ensure ascending
    use_rounds_oob = nbp_basic['use_rounds'][np.where((nbp_basic['use_rounds'] < 0) |
                                                      (nbp_basic['use_rounds'] >= n_rounds))[0]]
    if use_rounds_oob.size > 0:
        raise errors.OutOfBoundsError("use_rounds", use_rounds_oob[0], 0, n_rounds-1)
    # load in metadata of nd2 file corresponding to first round
    first_round_raw = os.path.join(config_file['input_dir'], config_file['round'][0]+config_file['raw_extension'])
    metadata = iss.utils.nd2.get_metadata(first_round_raw)
    # get tile info
    tile_sz = metadata['sizes']['x']
    n_tiles = metadata['sizes']['t']
    if config_basic['use_tiles'] is None:
        nbp_basic['use_tiles'] = list(np.arange(n_tiles))
    nbp_basic['use_tiles'].sort()
    use_tiles_oob = nbp_basic['use_tiles'][np.where((nbp_basic['use_tiles'] < 0) |
                                                      (nbp_basic['use_tiles'] >= n_tiles))[0]]
    if use_tiles_oob.size > 0:
        raise errors.OutOfBoundsError("use_tiles", use_tiles_oob[0], 0, n_tiles-1)
    # get channel info
    n_channels = metadata['sizes']['c']
    if config_basic['use_channels'] is None:
        nbp_basic['use_channels'] = list(np.arange(n_channels))
    nbp_basic['use_channels'].sort()
    use_channels_oob = nbp_basic['use_channels'][np.where((nbp_basic['use_channels'] < 0) |
                                                          (nbp_basic['use_channels'] >= n_channels))[0]]
    if use_channels_oob.size > 0:
        raise errors.OutOfBoundsError("use_channels", use_channels_oob[0], 0, n_channels - 1)
    # get z info
    if config_basic['use_z'] is None:
        nbp_basic['use_z'] = list(np.arange(metadata['sizes']['z']))
    if config_basic['ignore_first_z_plane'] and 0 in nbp_basic['use_z']:
        nbp_basic['use_z'].remove(0)
    nbp_basic['use_z'].sort()
    nz = len(nbp_basic['use_z'])  # number of z planes in tiff file (not necessarily the same as in nd2)
    use_z_oob = nbp_basic['use_z'][np.where((nbp_basic['use_z'] < 0) |
                                                          (nbp_basic['use_z'] >= metadata['sizes']['z']))[0]]
    if use_z_oob.size > 0:
        raise errors.OutOfBoundsError("use_z", use_z_oob[0], 0, metadata['sizes']['z'] - 1)

    tilepos_yx_nd2, tilepos_yx = get_tilepos(metadata['xy_pos'], tile_sz)
    nbp_basic['use_anchor'] = False
    if config_file['anchor'] is not None:
        # always have anchor as first round after imaging rounds
        nbp_basic['anchor_round'] = n_rounds
        if config_basic['anchor_channel'] is not None:
            if not 0 <= config_basic['anchor_channel'] <= n_channels-1:
                raise errors.OutOfBoundsError("anchor_channel", config_basic['anchor_channel'], 0, n_channels-1)
            nbp_basic['ref_round'] = nbp_basic['anchor_round']
            nbp_basic['ref_channel'] = config_basic['anchor_channel']
            nbp_basic['use_anchor'] = True
            warnings.warn("Anchor file given and anchor channel specified - will use anchor round")
        else:
            warnings.warn("Anchor file given but anchor channel not specified - will not use anchor round")
        round_files = config_file['round'] + [config_file['anchor']]
        nbp_basic['n_extra_rounds'] = 1
    else:
        nbp_basic['anchor_round'] = None
        round_files = config_file['round']
        nbp_file['anchor'] = None
        nbp_basic['n_extra_rounds'] = 0
        if not 0 <= nbp_basic['ref_round'] <= n_rounds - 1:
            raise errors.OutOfBoundsError("ref_round", nbp_basic['ref_round'], 0, n_rounds-1)
        warnings.warn("Anchor file not given - will not use anchor round")

    if not 0 <= nbp_basic['ref_channel'] <= n_channels - 1:
        raise errors.OutOfBoundsError("ref_channel", nbp_basic['ref_channel'], 0, n_channels-1)
    if config_basic['3d']:
        tile_names = get_tile_file_names(config_file['tile_dir'], round_files, tilepos_yx_nd2,
                                         config_file['matlab_tile_names'], n_channels)
    else:
        tile_names = get_tile_file_names(config_file['tile_dir'], round_files, tilepos_yx_nd2,
                                         config_file['matlab_tile_names'])

    nbp_file['tile'] = tile_names.tolist()  # tiff tile file paths list [n_tiles x n_rounds (x n_channels if 3D)]
    nbp_basic['n_rounds'] = n_rounds  # int, number of imaging rounds
    nbp_basic['tile_sz'] = tile_sz  # xy dimension of tiles in pixels.
    nbp_basic['n_tiles'] = n_tiles  # int, number of tiles
    nbp_basic['n_channels'] = n_channels  # int, number of imaging channels
    nbp_basic['nz'] = nz  # number of z-planes used to make tiff images
    nbp_basic['tilepos_yx_nd2'] = tilepos_yx_nd2  # numpy array, yx coordinate of tile with nd2 index.
    nbp_basic['tilepos_yx'] = tilepos_yx  # and with tiff index
    nbp_basic['pixel_size_xy'] = metadata['pixel_microns']  # pixel size in microns in xy
    nbp_basic['pixel_size_z'] = metadata['pixel_microns_z']  # and z directions.
    return nbp_file, nbp_basic
