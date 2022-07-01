import numpy as np
import os
from .. import utils, setup
import warnings
from datetime import datetime
from ..setup.notebook import NotebookPage
from typing import Tuple


def set_basic_info(config_file: dict, config_basic: dict) -> Tuple[NotebookPage, NotebookPage]:
    """
    Adds info from `'file_names'` and `'basic_info'` sections of config file
    to notebook pages of the same name.

    To `file_names` page, the following is also added:
    `tile`, `big_dapi_image`, `big_anchor_image`

    To `basic_info` page, the following is also added:
    `anchor_round`, `n_rounds`, `n_extra_rounds`, `n_tiles`, `n_channels`, `nz`, `tile_sz`, `tilepos_yx`,
    `tilepos_yx_nd2`, `pixel_size_xy`, `pixel_size_z`, `tile_centre`, `use_anchor`.

    See `'file_names'` and `'basic_info'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        config_file: Dictionary obtained from `'file_names'` section of config file.
        config_basic: Dictionary obtained from `'basic_info'` section of config file.

    Returns:
        - `NotebookPage[file_names]` - Page contains all files that are used throughout the pipeline.
        - `NotebookPage[basic_info]` - Page contains information that is used at all stages of the pipeline.
    """
    # remove file extension from round and anchor file names if it is present
    if config_file['round'] is None:
        if config_file['anchor'] is None:
            raise ValueError(f'Neither imaging rounds nor anchor_round provided')
        config_file['round'] = []  # Sometimes the case where just want to run the anchor round.
    config_file['round'] = [r.replace(config_file['raw_extension'], '') for r in config_file['round']]
    if config_file['anchor'] is not None:
        config_file['anchor'] = config_file['anchor'].replace(config_file['raw_extension'], '')
        if config_basic['anchor_channel'] is not None:
            config_basic['ref_round'] = None  # so can set to anchor round later
            config_basic['ref_channel'] = None  # so can set to anchor channel later
    else:
        config_basic['anchor_channel'] = None  # set anchor channel to None if no anchor round
        config_basic['dapi_channel'] = None  # set dapi channel to None if no anchor round

    # initialise log object with info from config
    nbp_file = setup.NotebookPage("file_names", config_file)
    nbp_basic = setup.NotebookPage("basic_info", config_basic)

    # add dapi channel and anchor channel to notebook even if set to None.
    if config_basic['dapi_channel'] is None:
        nbp_basic.dapi_channel = None
        nbp_file.big_dapi_image = None
    else:
        nbp_file.big_dapi_image = os.path.join(config_file['output_dir'], 'dapi_image.npz')
    if config_basic['anchor_channel'] is None:
        nbp_basic.anchor_channel = None
    nbp_file.big_anchor_image = os.path.join(config_file['output_dir'], 'anchor_image.npz')

    if config_file['dye_camera_laser'] is None:
        nbp_file.dye_camera_laser = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                 'dye_camera_laser_raw_intensity.csv')

    if config_file['psf'] is None:
        # where to save psf, indicating average spot shape in raw image
        if nbp_basic.is_3d:
            psf_file = os.path.join(config_file['output_dir'], 'psf.npy')
            if not os.path.isfile(psf_file):
                nbp_file.psf = psf_file
            else:
                # if file already exists, specify file with different name based on current time
                dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M")
                nbp_file.psf = os.path.join(config_file['output_dir'], 'psf_' + dt_string + '.npy')
        else:
            nbp_file.psf = None

    if config_file['omp_spot_shape'] is None:
        # where to save omp_spot_shape, indicating average spot shape in omp coefficient sign images.
        omp_spot_shape_file = os.path.join(config_file['output_dir'], 'omp_spot_shape.npy')
        if not os.path.isfile(omp_spot_shape_file):
            nbp_file.omp_spot_shape = omp_spot_shape_file
        else:
            # if file already exists, specify file with different name based on current time
            dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M")
            nbp_file.omp_spot_shape = os.path.join(config_file['output_dir'], 'omp_spot_shape_' + dt_string + '.npy')

    # Add files so save omp results after each tile as security if hit any bugs
    omp_spot_info_file = os.path.join(config_file['output_dir'], 'omp_spot_info.npy')
    omp_spot_coef_file = os.path.join(config_file['output_dir'], 'omp_spot_coef.npz')
    if os.path.isfile(omp_spot_info_file) or os.path.isfile(omp_spot_coef_file):
        dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M")
        nbp_file.omp_spot_info = os.path.join(config_file['output_dir'], 'omp_spot_info_' + dt_string + '.npy')
        nbp_file.omp_spot_coef = os.path.join(config_file['output_dir'], 'omp_spot_coef_' + dt_string + '.npz')
    else:
        nbp_file.omp_spot_info = omp_spot_info_file
        nbp_file.omp_spot_coef = omp_spot_coef_file

    # get round info from config file
    n_rounds = len(config_file['round'])
    if config_basic['use_rounds'] is None:
        nbp_basic.use_rounds = list(np.arange(n_rounds))
    nbp_basic.use_rounds.sort()  # ensure ascending
    use_rounds_oob = [val for val in nbp_basic.use_rounds if val < 0 or val >= n_rounds]
    if len(use_rounds_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_rounds", use_rounds_oob[0], 0, n_rounds-1)
    # load in metadata of nd2 file corresponding to first round
    # Test for number of rounds in case of separate round registration and load metadata
    # from anchor round in that case
    if len(config_file['round']) > 0:
        first_round_raw = os.path.join(config_file['input_dir'], config_file['round'][0]+config_file['raw_extension'])
    else:
        first_round_raw = os.path.join(config_file['input_dir'], config_file['anchor'] + config_file['raw_extension'])

    metadata = utils.nd2.get_metadata(first_round_raw)
    # get tile info
    tile_sz = metadata['sizes']['x']
    n_tiles = metadata['sizes']['t']
    if config_basic['use_tiles'] is None:
        nbp_basic.use_tiles = list(np.arange(n_tiles))
    nbp_basic.use_tiles.sort()
    use_tiles_oob = [val for val in nbp_basic.use_tiles if val < 0 or val >= n_tiles]
    if len(use_tiles_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_tiles", use_tiles_oob[0], 0, n_tiles-1)
    # get channel info
    n_channels = metadata['sizes']['c']
    if config_basic['use_channels'] is None:
        nbp_basic.use_channels = list(np.arange(n_channels))
    nbp_basic.use_channels.sort()
    use_channels_oob = [val for val in nbp_basic.use_channels if val < 0 or val >= n_channels]
    if len(use_channels_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_channels", use_channels_oob[0], 0, n_channels - 1)
    # get z info
    if config_basic['use_z'] is None:
        nbp_basic.use_z = list(np.arange(metadata['sizes']['z']))
    elif len(config_basic['use_z']) == 2:
        # use consecutive values if only 2 given.
        nbp_basic.use_z[:] = list(np.arange(config_basic['use_z'][0], config_basic['use_z'][1] + 1))
    if config_basic['ignore_first_z_plane'] and 0 in nbp_basic.use_z:
        nbp_basic.use_z.remove(0)
    nbp_basic.use_z.sort()
    nz = len(nbp_basic.use_z)  # number of z planes in npy file (not necessarily the same as in nd2)
    use_z_oob = [val for val in nbp_basic.use_z if val < 0 or val >= metadata['sizes']['z']]
    if len(use_z_oob) > 0:
        raise utils.errors.OutOfBoundsError("use_z", use_z_oob[0], 0, metadata['sizes']['z'] - 1)

    # get dye info
    if config_basic['dye_names'] is None:
        nbp_basic.dye_names = None
        warnings.warn(f"dye_names not specified so assuming separate dye for each channel.")
        n_dyes = n_channels
    else:
        n_dyes = len(config_basic['dye_names'])
    if config_basic['use_dyes'] is None:
        if config_basic['dye_names'] is None:
            nbp_basic.use_dyes = nbp_basic.use_channels
        else:
            nbp_basic.use_dyes = list(np.arange(n_dyes))
    if config_basic['channel_camera'] is None:
        nbp_basic.channel_camera = None
    if config_basic['channel_laser'] is None:
        nbp_basic.channel_laser = None

    tilepos_yx_nd2, tilepos_yx = setup.get_tilepos(metadata['xy_pos'], tile_sz)
    use_anchor = False
    if config_file['anchor'] is not None:
        # always have anchor as first round after imaging rounds
        nbp_basic.anchor_round = n_rounds
        if config_basic['anchor_channel'] is not None:
            if not 0 <= config_basic['anchor_channel'] <= n_channels-1:
                raise utils.errors.OutOfBoundsError("anchor_channel", config_basic['anchor_channel'], 0, n_channels-1)
            nbp_basic.ref_round = nbp_basic.anchor_round
            nbp_basic.ref_channel = config_basic['anchor_channel']
            use_anchor = True
            warnings.warn(f"Anchor file given and anchor channel specified."
                          f"\nWill use anchor round, channel {nbp_basic.ref_channel} as reference")
        elif nbp_basic.ref_round == nbp_basic.anchor_round:
            warnings.warn(f"Anchor file given but anchor channel not specified."
                          f"\nWill use anchor round, channel {nbp_basic.ref_channel} as reference")
        else:
            warnings.warn(f"Anchor file given but anchor channel not specified."
                          f"\nWill use round {nbp_basic.ref_round} (not anchor),"
                          f" channel {nbp_basic.ref_channel} as reference")
        round_files = config_file['round'] + [config_file['anchor']]
        nbp_basic.n_extra_rounds = 1
    else:
        nbp_basic.anchor_round = None
        round_files = config_file['round']
        nbp_file.anchor = None
        nbp_basic.n_extra_rounds = 0
        if not 0 <= nbp_basic.ref_round <= n_rounds - 1:
            raise utils.errors.OutOfBoundsError("ref_round", nbp_basic.ref_round, 0, n_rounds-1)
        warnings.warn(f"Anchor file not given."
                      f"\nWill use round {nbp_basic.ref_round}, channel {nbp_basic.ref_channel} as reference")

    if not 0 <= nbp_basic.ref_channel <= n_channels - 1:
        raise utils.errors.OutOfBoundsError("ref_channel", nbp_basic.ref_channel, 0, n_channels-1)

    if config_basic['is_3d']:
        tile_names = setup.get_tile_file_names(config_file['tile_dir'], round_files, n_tiles, n_channels)
    else:
        tile_names = setup.get_tile_file_names(config_file['tile_dir'], round_files, n_tiles)

    nbp_file.tile = tile_names.tolist()  # npy tile file paths list [n_tiles x n_rounds (x n_channels if 3D)]
    nbp_basic.n_rounds = n_rounds  # int, number of imaging rounds
    nbp_basic.tile_sz = tile_sz  # xy dimension of tiles in pixels.
    nbp_basic.n_tiles = n_tiles  # int, number of tiles
    nbp_basic.n_channels = n_channels  # int, number of imaging channels
    nbp_basic.nz = nz  # number of z-planes used to make npy images
    nbp_basic.n_dyes = n_dyes  # int, number of dyes
    # subtract tile_centre from local pixel coordinates to get centered local tile coordinates
    if not nbp_basic.is_3d:
        nz = 1
    nbp_basic.tile_centre = (np.array([tile_sz, tile_sz, nz]) - 1) / 2
    nbp_basic.tilepos_yx_nd2 = tilepos_yx_nd2  # numpy array, yx coordinate of tile with nd2 index.
    nbp_basic.tilepos_yx = tilepos_yx  # and with npy index
    nbp_basic.pixel_size_xy = metadata['pixel_microns']  # pixel size in microns in xy
    nbp_basic.pixel_size_z = metadata['pixel_microns_z']  # and z directions.
    nbp_basic.use_anchor = use_anchor
    return nbp_file, nbp_basic
