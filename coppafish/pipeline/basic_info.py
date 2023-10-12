import numpy as np
import os
from .. import utils, setup
import warnings
from ..setup.notebook import NotebookPage
import json


def set_basic_info(config_file: dict, config_basic: dict, n_rounds: int = 7) -> NotebookPage:
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
        n_rounds: in order to accommodate with new file format, the number of rounds is specified (default=7)

    Returns:
        - `NotebookPage[basic_info]` - Page contains information that is used at all stages of the pipeline.
    """
    nbp = NotebookPage('basic_info')
    nbp.is_3d = config_basic['is_3d']

    # TODO: Get rid of this
    # First condition refers to situation where jobs not used, alternative if jobs is used
    if len(config_file['round']) == n_rounds:
        n_rounds = len(config_file['round'])
    else:
        # In this case we are using jobs format
        all_files = os.listdir(config_file['input_dir'])
        all_files.sort()
        lasers = utils.nd2.get_jobs_lasers(config_file['input_dir'], all_files)
        n_lasers = len(lasers)
        n_tiles, n_rounds = utils.nd2.get_jobs_rounds_tiles(config_file['input_dir'], all_files, n_lasers)

    # TODO: Get rid of ref round/channel
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

    # These 3 are different ways of obtaining the file from which we will obtain the metadata
    if config_file['round'] is not None:
        if len(config_file) > 0:
            first_round_raw = os.path.join(config_file['input_dir'], config_file['round'][0])
        else:
            first_round_raw = os.path.join(config_file['input_dir'], config_file['anchor'])
    else:
        first_round_raw = os.path.join(config_file['input_dir'], all_files[0])

    if config_file['raw_extension'] == '.nd2':
        # load in metadata of nd2 file corresponding to first round
        # Test for number of rounds in case of separate round registration and load metadata
        # from anchor round in that case
        metadata = utils.nd2.get_metadata(first_round_raw + config_file['raw_extension'])
        n_tiles = metadata['sizes']['t']

    elif config_file['raw_extension'] == '.npy':
        # Load in metadata as dictionary from a json file
        config_file['raw_metadata'] = config_file['raw_metadata'].replace('.json', '')
        metadata_file = os.path.join(config_file['input_dir'], config_file['raw_metadata'] + '.json')
        metadata = json.load(open(metadata_file))

        # Check metadata info matches that in first round npy file.
        use_tiles_nd2 = utils.raw.metadata_sanity_check(metadata, first_round_raw)
        n_tiles = metadata['sizes']['t']

    elif config_file['raw_extension'] == 'jobs':
        # Get the basic metadata from the first file
        # Then iterate over every file of the first round to extract xy_pos
        metadata = utils.nd2.get_metadata(first_round_raw + '.nd2')
        all_files = os.listdir(config_file['input_dir'])
        all_files.sort()
        n_tiles = int(len(all_files) / 7 / (n_rounds + 1))
        first_round_files = all_files[::7][:n_tiles]
        metadata['sizes']['t'] = n_tiles
        metadata['xy_pos'] = utils.nd2.get_jobs_xypos(config_file['input_dir'], first_round_files)
    else:
        raise ValueError(f"config_file['raw_extension'] should be either '.nd2' or '.npy' but it is "
                         f"{config_file['raw_extension']}.")

    # get channel info only for nd2 or npy files
    if config_file['raw_extension'] != 'jobs':
        n_channels = metadata['sizes']['c']
        if config_basic['use_channels'] is None:
            config_basic['use_channels'] = list(np.arange(n_channels))
        nbp.use_channels = config_basic['use_channels']
        nbp.use_channels.sort()
        use_channels_oob = [val for val in nbp.use_channels if val < 0 or val >= n_channels]
        if len(use_channels_oob) > 0:
            raise utils.errors.OutOfBoundsError("use_channels", use_channels_oob[0], 0, n_channels - 1)
    elif config_file['raw_extension'] == 'jobs':
        n_channels = metadata['sizes']['c'] * 7
        nbp.use_channels = config_basic['use_channels']
        nbp.use_channels.sort()

    # # get channel info
    # # n_channels = metadata['sizes']['c']
    # # if config_basic['use_channels'] is None:
    # #     config_basic['use_channels'] = list(np.arange(n_channels))
    # nbp.use_channels = config_basic['use_channels']
    # nbp.use_channels.sort()
    # use_channels_oob = [val for val in nbp.use_channels if val < 0 or val >= n_channels]
    # if len(use_channels_oob) > 0:
    #     raise utils.errors.OutOfBoundsError("use_channels", use_channels_oob[0], 0, n_channels - 1)
    #

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
    tilepos_yx_nd2, tilepos_yx = setup.get_tilepos(np.asarray(metadata['xy_pos']), tile_sz,
                                                   format='new')
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


def set_basic_info_new(config: dict) -> NotebookPage:
    """
    Adds info from `'basic_info'` section of config file to notebook page.

    To `basic_info` page, the following is also added:
    `anchor_round`, `n_rounds`, `n_extra_rounds`, `n_tiles`, `n_channels`, `nz`, `tile_sz`, `tilepos_yx`,
    `tilepos_yx_nd2`, `pixel_size_xy`, `pixel_size_z`, `tile_centre`, `use_anchor`.

    See `'basic_info'` sections of `notebook_comments.json` file
    for description of the variables.

    Args:
        - `config` : `dict` - Config dictionary.
    Returns:
        - `NotebookPage[basic_info]` - Page contains information that is used at all stages of the pipeline.
    """
    # Initialize Notebook
    nbp = NotebookPage('basic_info')

    # Now break the page contents up into 2 types, contents that must be read in from the config and those that can
    # be computed from the metadata
    config_file = config['file_names']
    config_basic = config['basic_info']

    # Stage 1: Compute metadata. This is done slightly differently in the 3 cases of different raw extensions
    raw_extension = utils.nd2.get_raw_extension(config_file['input_dir'])
    all_files = []
    for root, directories, filenames in os.walk(config_file['input_dir']):
        for filename in filenames:
            all_files.append(os.path.join(root, filename))
    all_files.sort()
    if raw_extension == '.nd2':
        if config_file['round'] is None and config_file['anchor'] is None:
            raise ValueError(f"config_file['round'] or config_file['anchor'] should not both be left blank")
        # load in metadata of nd2 file corresponding to first round
        # Allow for degenerate case when only anchor has been provided
        if config_file['round'] is not None:
            first_round_raw = os.path.join(config_file['input_dir'], config_file['round'][0])
        else:
            first_round_raw = os.path.join(config_file['input_dir'], config_file['anchor'])
        metadata = utils.nd2.get_metadata(first_round_raw + raw_extension, config=config)

    elif raw_extension == '.npy':
        # Load in metadata as dictionary from a json file
        metadata_file = [file for file in all_files if file.endswith('.json')][0]
        if metadata_file is None:
            raise ValueError(f"There is no json metadata file in input_dir. This should have been set at the point of "
                             f"ND2 extraction to npy.")
        metadata = json.load(open(metadata_file))

    elif raw_extension == 'jobs':
        metadata = utils.nd2.get_jobs_metadata(all_files, config_file['input_dir'], config=config)
    else:
        raise ValueError(f"config_file['raw_extension'] should be either '.nd2' or '.npy' but it is "
                         f"{config_file['raw_extension']}.")

    # Stage 2: Read in page contents from config that cannot be computed from metadata.
    # the metadata. First 12 keys in the basic info page are only variables that the user can influence
    for key, value in list(config_basic.items())[:12]:
        nbp.__setattr__(key=key, value=value)

    # Only 3 of these can NOT be left empty
    if nbp.dye_names is None or nbp.tile_pixel_value_shift is None or nbp.use_anchor is None:
        raise ValueError('One or more of the 3 variables which cannot be computed from anything else has been left '
                         'empty. Please fill in the use_anchor, dye_names and pixel_value_shift variables.')

    # Stage 3: Fill in all the metadata except the last item, xy_pos
    for key, value in metadata.items():
        if key != 'xy_pos' and key != 'nz':
            nbp.__setattr__(key=key, value=value)

    # Stage 4: If anything from the first 12 entries has been left blank, deal with that here.
    # Unfortunately, this is just many if statements as all blank entries need to be handled differently.
    # Notebook doesn't allow us to reset a value once it has been set so must delete and reset.

    # Next condition just says that if we are using the anchor and we don't specify the anchor round we will default it
    # to the final round. Add an extra round for the anchor and reduce the number of non anchor rounds by 1.
    if nbp.use_anchor:
        # Tell software that extra round is just an extra round and reduce the number of rounds
        nbp.n_extra_rounds = 1
        if nbp.anchor_round is None:
            del nbp.anchor_round
            nbp.anchor_round = metadata['n_rounds']
        if nbp.anchor_channel is None:
            raise ValueError('Need to provide an anchor channel if using anchor!')
    else:
        nbp.n_extra_rounds = 0

    # If no use_tiles given, default to all
    if nbp.use_tiles is None:
        del nbp.use_tiles
        nbp.use_tiles = np.arange(metadata['n_tiles']).tolist()

    # If no use_rounds given, replace none with [], unless non jobs and user has provided the rounds in the file names
    if nbp.use_rounds is None:
        del nbp.use_rounds
        if config_file['round'] is not None and raw_extension != 'jobs':
            nbp.use_rounds = np.arange(len(config_file['round'])).tolist()
        else:
            nbp.use_rounds = np.arange(0, nbp.n_rounds).tolist()

    if nbp.use_channels is None:
        del nbp.use_channels
        nbp.use_channels = np.arange(metadata['n_channels']).tolist()

    # If no use_z given, default to all except the first if ignore_first_z_plane = True
    if nbp.use_z is None:
        del nbp.use_z
        nbp.use_z = np.arange(int(config_basic['ignore_first_z_plane']), metadata['nz']).tolist()
    if nbp.nz is None:
        # This has not been assigned yet but now we can be sure that use_z not None!
        nbp.nz = metadata['nz']

    if nbp.use_dyes is None:
        del nbp.use_dyes
        nbp.use_dyes = np.arange(len(nbp.dye_names)).tolist()
        nbp.n_dyes = len(nbp.use_dyes)

    # Run into annoying problem of tilepos_yx, tilepos_yx_nd2 and tile_centre being saved as lists if they are loaded
    # from json metadata as json cannot save npys.
    if raw_extension == '.npy':
        del nbp.tilepos_yx, nbp.tilepos_yx_nd2, nbp.tile_centre
        nbp.tilepos_yx = np.array(metadata['tilepos_yx'])
        nbp.tilepos_yx_nd2 = np.array(metadata['tilepos_yx_nd2'])
        nbp.tile_centre = np.array(metadata['tile_centre'])

    # If preseq round is a file, set pre_seq_round to True, else False and raise warning that pre_seq_round is not a
    # file
    if config_file['pre_seq'] is None:
        nbp.use_preseq = False
        nbp.pre_seq_round = None
        # Return here as we don't need to check if the file exists
        return nbp

    if raw_extension == '.nd2':
        if os.path.isfile(os.path.join(config_file['input_dir'], config_file['pre_seq'] + '.nd2')):
            nbp.use_preseq = True
            nbp.pre_seq_round = nbp.anchor_round + 1
            n_extra_rounds = nbp.n_extra_rounds
            del nbp.n_extra_rounds
            nbp.n_extra_rounds = n_extra_rounds + 1
        else:
            nbp.use_preseq = False
            nbp.pre_seq_round = None
            warnings.warn(f"Pre-sequencing round not found at "
                          f"{os.path.join(config_file['input_dir'], config_file['pre_seq'] +'.nd2')}. "
                          f"Setting pre_seq_round to False. If this is not what you want, please check that the "
                          f"pre_seq_round variable in the config file is set to the correct file name.")

    elif raw_extension == '.npy':
        if os.path.isdir(os.path.join(config_file['input_dir'], config_file['pre_seq'])):
            nbp.use_preseq = True
            nbp.pre_seq_round = nbp.anchor_round + 1
            n_extra_rounds = nbp.n_extra_rounds
            del nbp.n_extra_rounds
            nbp.n_extra_rounds = n_extra_rounds + 1
        else:
            nbp.use_preseq = False
            nbp.pre_seq_round = None
            warnings.warn(f"Pre-sequencing round not found at "
                          f"{os.path.join(config_file['input_dir'], config_file['pre_seq'] +'.nd2')}. "
                          f"Setting pre_seq_round to False. If this is not what you want, please check that the "
                          f"pre_seq_round variable in the config file is set to the correct file name.")

    elif raw_extension == 'jobs':
        if config_file['pre_seq']:
            nbp.use_preseq = True
            nbp.pre_seq_round = nbp.anchor_round + 1
            n_extra_rounds = nbp.n_extra_rounds
            del nbp.n_extra_rounds
            nbp.n_extra_rounds = n_extra_rounds + 1
        else:
            nbp.use_preseq = False
            nbp.pre_seq_round = None
            warnings.warn(f"Pre-sequencing round not found at "
                          f"{config['input_dir']}."
                          f"Setting pre_seq_round to False. If this is not what you want, please check that the "
                          f"pre_seq_round variable in the config file is set to the correct file name.")

    return nbp
