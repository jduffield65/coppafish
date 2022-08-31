import numpy as np
import os
from .. import utils, setup
import warnings
from ..setup.notebook import NotebookPage
import json


def set_basic_info(config_file: dict, config_basic: dict) -> NotebookPage:
    """
    Adds info from `'basic_info'` section of config file to notebook page.

    To `basic_info` page, the following is also added:
    `anchor_round`, `n_rounds`, `n_extra_rounds`, `n_tiles`, `n_channels`, `nz`, `tile_sz`, `tilepos_yx`,
    `tilepos_yx_nd2`, `pixel_size_xy`, `pixel_size_z`, `tile_centre`, `use_anchor`.

    See `'basic_info'` sections of `notebook_comments.json` file
    for description of the variables.

    Args:
        config_file: Dictionary obtained from `'file_names'` section of config file.
        config_basic: Dictionary obtained from `'basic_info'` section of config file.

    Returns:
        - `NotebookPage[basic_info]` - Page contains information that is used at all stages of the pipeline.
    """
    nbp = NotebookPage('basic_info')
    nbp.is_3d = config_basic['is_3d']

    # Deal with case where no imaging rounds, just want to run anchor round.
    if config_file['round'] is None:
        if config_file['anchor'] is None:
            raise ValueError(f'Neither imaging rounds nor anchor_round provided')
        config_file['round'] = []
    n_rounds = len(config_file['round'])

    # Set ref/anchor round/channel
    if config_file['anchor'] is None:
        config_basic['anchor_channel'] = None  # set anchor channel to None if no anchor round
        config_basic['dapi_channel'] = None  # set dapi channel to None if no anchor round
        if config_basic['ref_round'] is None:
            raise ValueError('No anchor round used, but ref_round not specified')
        if config_basic['ref_channel'] is None:
            raise ValueError('No anchor round used, but ref_channel not specified')
        nbp.anchor_round = None
        nbp.n_extra_rounds = 0
        nbp.use_anchor = False
        warnings.warn(f"Anchor file not given."
                      f"\nWill use round {config_basic['ref_round']}, "
                      f"channel {config_basic['ref_channel']} as reference")
    else:
        if config_basic['anchor_channel'] is None:
            raise ValueError('Using anchor round, but anchor_channel not specified')
        # always have anchor as first round after imaging rounds
        nbp.anchor_round = n_rounds
        config_basic['ref_round'] = nbp.anchor_round
        config_basic['ref_channel'] = config_basic['anchor_channel']
        nbp.n_extra_rounds = 1
        nbp.use_anchor = True
        warnings.warn(f"Anchor file given and anchor channel specified."
                      f"\nWill use anchor round, channel {config_basic['anchor_channel']} as reference")
    nbp.anchor_channel = config_basic['anchor_channel']
    nbp.dapi_channel = config_basic['dapi_channel']
    nbp.ref_round = config_basic['ref_round']
    nbp.ref_channel = config_basic['ref_channel']

    if config_basic['use_rounds'] is None:
        config_basic['use_rounds'] = list(np.arange(n_rounds))
    nbp.use_rounds = config_basic['use_rounds']
    nbp.use_rounds.sort()  # ensure ascending
    use_rounds_oob = [val for val in nbp.use_rounds if val < 0 or val >= n_rounds]
    if len(use_rounds_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_rounds", use_rounds_oob[0], 0, n_rounds - 1)

    if len(config_file['round']) > 0:
        first_round_raw = os.path.join(config_file['input_dir'], config_file['round'][0])
    else:
        first_round_raw = os.path.join(config_file['input_dir'], config_file['anchor'])
    if config_file['raw_extension'] == '.nd2':
        # load in metadata of nd2 file corresponding to first round
        # Test for number of rounds in case of separate round registration and load metadata
        # from anchor round in that case
        metadata = utils.nd2.get_metadata(first_round_raw + config_file['raw_extension'])
    elif config_file['raw_extension'] == '.npy':
        # Load in metadata as dictionary from a json file
        config_file['raw_metadata'] = config_file['raw_metadata'].replace('.json', '')
        metadata_file = os.path.join(config_file['input_dir'], config_file['raw_metadata'] + '.json')
        metadata = json.load(open(metadata_file))
        # Check metadata info matches that in first round npy file.
        use_tiles_nd2 = utils.raw.metadata_sanity_check(metadata, first_round_raw)
    else:
        raise ValueError(f"config_file['raw_extension'] should be either '.nd2' or '.npy' but it is "
                         f"{config_file['raw_extension']}.")



    # get channel info
    n_channels = metadata['sizes']['c']
    if config_basic['use_channels'] is None:
        config_basic['use_channels'] = list(np.arange(n_channels))
    nbp.use_channels = config_basic['use_channels']
    nbp.use_channels.sort()
    use_channels_oob = [val for val in nbp.use_channels if val < 0 or val >= n_channels]
    if len(use_channels_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_channels", use_channels_oob[0], 0, n_channels - 1)

    # get z info
    if config_basic['use_z'] is None:
        config_basic['use_z'] = list(np.arange(metadata['sizes']['z']))
    elif len(config_basic['use_z']) == 2:
        # use consecutive values if only 2 given.
        config_basic['use_z'] = list(np.arange(config_basic['use_z'][0], config_basic['use_z'][1] + 1))
    if config_basic['ignore_first_z_plane'] and 0 in config_basic['use_z']:
        config_basic['use_z'].remove(0)
    nbp.use_z = config_basic['use_z']
    nbp.use_z.sort()
    use_z_oob = [val for val in nbp.use_z if val < 0 or val >= metadata['sizes']['z']]
    if len(use_z_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_z", use_z_oob[0], 0, metadata['sizes']['z'] - 1)

    # get tile info
    tile_sz = metadata['sizes']['x']
    n_tiles = metadata['sizes']['t']
    tilepos_yx_nd2, tilepos_yx = setup.get_tilepos(np.asarray(metadata['xy_pos']), tile_sz)
    nbp.tilepos_yx_nd2 = tilepos_yx_nd2  # numpy array, yx coordinate of tile with nd2 index.
    nbp.tilepos_yx = tilepos_yx  # and with npy index

    if config_file['raw_extension'] == '.npy':
        # Read tile indices from raw data folder and set to use_tiles if not specified already.
        use_tiles_folder = utils.npy.get_npy_tile_ind(use_tiles_nd2, tilepos_yx_nd2, tilepos_yx)
        if config_basic['use_tiles'] is None:
            config_basic['use_tiles'] = use_tiles_folder
        elif np.setdiff1d(config_basic['use_tiles'], use_tiles_folder).size > 0:
            raise ValueError(f"config_basic['use_tiles'] = {config_basic['use_tiles']}\n"
                             f"But in the folder:\n{first_round_raw}\nTiles Available are {use_tiles_folder}.")
    if config_basic['use_tiles'] is None:
        config_basic['use_tiles'] = list(np.arange(n_tiles))
    if config_basic['ignore_tiles'] is not None:
        config_basic['use_tiles'] = list(np.setdiff1d(config_basic['use_tiles'], config_basic['ignore_tiles']))
    nbp.use_tiles = config_basic['use_tiles']
    nbp.use_tiles.sort()
    use_tiles_oob = [val for val in nbp.use_tiles if val < 0 or val >= n_tiles]
    if len(use_tiles_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_tiles", use_tiles_oob[0], 0, n_tiles - 1)

    # get dye info
    if config_basic['dye_names'] is None:
        warnings.warn(f"dye_names not specified so assuming separate dye for each channel.")
        n_dyes = n_channels
    else:
        # Ensure channel_camera/channel_laser are correct sizes
        n_dyes = len(config_basic['dye_names'])
        if config_basic['channel_camera'] is None:
            raise ValueError('dye_names specified but channel_camera is not.')
        elif len(config_basic['channel_camera']) != n_channels:
            raise ValueError(f"channel_camera contains {len(config_basic['channel_camera'])} values.\n"
                             f"But there must be a value for each channel and there are {n_channels} channels.")
        if config_basic['channel_laser'] is None:
            raise ValueError('dye_names specified but channel_laser is not.')
        elif len(config_basic['channel_laser']) != n_channels:
            raise ValueError(f"channel_laser contains {len(config_basic['channel_camera'])} values.\n"
                             f"But there must be a value for each channel and there are {n_channels} channels.")

    if config_basic['use_dyes'] is None:
        if config_basic['dye_names'] is None:
            config_basic['use_dyes'] = nbp.use_channels
        else:
            config_basic['use_dyes'] = list(np.arange(n_dyes))
    nbp.use_dyes = config_basic['use_dyes']
    nbp.dye_names = config_basic['dye_names']
    nbp.channel_camera = config_basic['channel_camera']
    nbp.channel_laser = config_basic['channel_laser']

    nbp.tile_pixel_value_shift = config_basic['tile_pixel_value_shift']

    # Add size info obtained from raw metadata to notebook page
    nbp.n_rounds = n_rounds  # int, number of imaging rounds
    nbp.tile_sz = tile_sz  # xy dimension of tiles in pixels.
    nbp.n_tiles = n_tiles  # int, number of tiles
    nbp.n_channels = n_channels  # int, number of imaging channels
    nbp.nz = len(nbp.use_z)  # number of z planes in npy file (not necessarily the same as in nd2)
    nbp.n_dyes = n_dyes  # int, number of dyes

    # subtract tile_centre from local pixel coordinates to get centered local tile coordinates
    if not nbp.is_3d:
        nz = 1
    else:
        nz = nbp.nz
    nbp.tile_centre = (np.array([tile_sz, tile_sz, nz]) - 1) / 2
    nbp.pixel_size_xy = metadata['pixel_microns']  # pixel size in microns in xy
    nbp.pixel_size_z = metadata['pixel_microns_z']  # and z directions.

    # Make sure reference rounds/channels are in range of data provided.
    if nbp.use_anchor:
        if not 0 <= nbp.ref_channel <= n_channels - 1:
            raise utils.errors.OutOfBoundsError("ref_channel", nbp.ref_channel, 0, n_channels - 1)
        if nbp.dapi_channel is not None:
            if not 0 <= nbp.dapi_channel <= n_channels - 1:
                raise utils.errors.OutOfBoundsError("dapi_channel", nbp.ref_channel, 0, n_channels - 1)
    else:
        # Seen as ref_channel is an imaging channel if anchor not used, ref_channel must be an imaging channels i.e.
        # must be in use_channels. Same goes for ref_round.
        if not np.isin(nbp.ref_channel, nbp.use_channels):
            raise ValueError(f"ref_channel is {nbp.ref_channel} which is not in use_channels = {nbp.use_channels}.")
        if not np.isin(nbp.ref_round, nbp.use_rounds):
            raise ValueError(f"ref_round is {nbp.ref_round} which is not in use_rounds = {nbp.use_rounds}.")

    return nbp
