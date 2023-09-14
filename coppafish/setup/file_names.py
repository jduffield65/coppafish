import os
from .tile_details import get_tile_file_names


def set_file_names(nb, nbp):
    """
    Function to set add `file_names` page to notebook. It requires notebook to be able to access a
    config file containing a `file_names` section and also the notebook to contain a `basic_info` page.

    !!! note
        This will be called every time the notebook is loaded to deal will case when `file_names` section of
        config file changed.

    Args:
        nb: *Notebook* containing at least the `basic_info` page.
        nbp: *NotebookPage* with no variables added.
            This is just to avoid initialising it within the function which would cause a circular import.

    """
    config = nb.get_config()['file_names']
    nbp.name = 'file_names'  # make sure name is correct
    # Copy some variables that are in config to page.
    nbp.input_dir = config['input_dir']
    nbp.output_dir = config['output_dir']
    nbp.tile_dir = config['tile_dir']
    nbp.fluorescent_bead_path = config['fluorescent_bead_path']

    # remove file extension from round and anchor file names if it is present
    if config['raw_extension'] == 'jobs':
        all_files = os.listdir(config['input_dir'])
        all_files.sort()  # Sort files by ascending number
        n_tiles = int(len(all_files)/7/8)
        config['round'] = [r.replace('.nd2', '') for r in all_files[:n_tiles*7*7]]
        config['anchor'] = [r.replace('.nd2', '') for r in all_files[n_tiles*7*7:]]

    else:
        if config['round'] is None:
            if config['anchor'] is None:
                raise ValueError(f'Neither imaging rounds nor anchor_round provided')
            config['round'] = []  # Sometimes the case where just want to run the anchor round.
        config['round'] = [r.replace(config['raw_extension'], '') for r in config['round']]

        if config['anchor'] is not None:
            config['anchor'] = config['anchor'].replace(config['raw_extension'], '')

    nbp.round = config['round']
    nbp.anchor = config['anchor']
    nbp.pre_seq_round = config['pre_seq_round']
    nbp.raw_extension = config['raw_extension']
    nbp.raw_metadata = config['raw_metadata']
    nbp.initial_bleed_matrix = config['initial_bleed_matrix']

    if nbp.initial_bleed_matrix is not None:
        assert os.path.isfile(nbp.initial_bleed_matrix), \
            f'Initial bleed matrix located at {nbp.initial_bleed_matrix} does not exist'

    if config['dye_camera_laser'] is None:
        # Default information is project
        config['dye_camera_laser'] = os.path.join(os.path.dirname(__file__), 'dye_camera_laser_raw_intensity.csv')
    nbp.dye_camera_laser = config['dye_camera_laser']

    if config['code_book'] is not None:
        config['code_book'] = config['code_book'].replace('.txt', '')
        nbp.code_book = config['code_book'] + '.txt'
    else:
        # If the user has not put their code_book in, default to the one included in this project
        config['code_book'] = os.path.join(os.getcwd(), 'coppafish/setup/code_book_73g.txt')

    # where to save scale and scale_anchor values used in extract step.
    config['scale'] = config['scale'].replace('.txt', '')
    nbp.scale = os.path.join(config['tile_dir'], config['scale'] + '.txt')

    # where to save psf, indicating average spot shape in raw image. Only ever needed in 3D.
    if nb.basic_info.is_3d:
        config['psf'] = config['psf'].replace('.npy', '')
        nbp.psf = os.path.join(config['output_dir'], config['psf'] + '.npy')
    else:
        nbp.psf = None

    # Add files to save find_spot results after each tile as security if hit any bugs
    config['spot_details_info'] = config['spot_details_info'].replace('.npy', '')
    nbp.spot_details_info = os.path.join(config['output_dir'], config['spot_details_info'] + '.npz')

    # where to save omp_spot_shape, indicating average spot shape in omp coefficient sign images.
    config['omp_spot_shape'] = config['omp_spot_shape'].replace('.npy', '')
    omp_spot_shape_file = os.path.join(config['output_dir'], config['omp_spot_shape'] + '.npy')
    nbp.omp_spot_shape = omp_spot_shape_file

    # Add files to save omp results after each tile as security if hit any bugs
    config['omp_spot_info'] = config['omp_spot_info'].replace('.npy', '')
    nbp.omp_spot_info = os.path.join(config['output_dir'], config['omp_spot_info'] + '.npy')
    config['omp_spot_coef'] = config['omp_spot_coef'].replace('.npz', '')
    nbp.omp_spot_coef = os.path.join(config['output_dir'], config['omp_spot_coef'] + '.npz')

    # Add files so save plotting information for pciseq
    config['pciseq'] = [val.replace('.csv', '') for val in config['pciseq']]
    nbp.pciseq = [os.path.join(config['output_dir'], val + '.csv') for val in config['pciseq']]

    # add dapi channel and anchor channel to notebook even if set to None.
    if config['big_dapi_image'] is None:
        nbp.big_dapi_image = None
    else:
        config['big_dapi_image'] = config['big_dapi_image'].replace('.npz', '')
        if nb.basic_info.dapi_channel is None:
            nbp.big_dapi_image = None
        else:
            nbp.big_dapi_image = os.path.join(config['output_dir'], config['big_dapi_image'] + '.npz')
    if config['big_anchor_image'] is None:
        nbp.big_anchor_image = None
    else:
        config['big_anchor_image'] = config['big_anchor_image'].replace('.npz', '')
        nbp.big_anchor_image = os.path.join(config['output_dir'], config['big_anchor_image'] + '.npz')

    if config['anchor'] is not None:
        round_files = config['round'] + [config['anchor']]
    else:
        round_files = config['round']

    if config['pre_seq_round'] is not None:
        round_files = round_files + [config['pre_seq_round']]

    if config['raw_extension'] == 'jobs':
        if nb.basic_info.is_3d:
            round_files = config['round'] + config['anchor']
            tile_names = get_tile_file_names(config['tile_dir'], round_files, nb.basic_info.n_tiles,
                                             nb.basic_info.n_channels, jobs=True)
        else:
            raise ValueError('JOBs file format is only compatible with 3D')
    else:
        if nb.basic_info.is_3d:
            tile_names = get_tile_file_names(config['tile_dir'], round_files, nb.basic_info.n_tiles,
                                             nb.basic_info.n_channels)
        else:
            tile_names = get_tile_file_names(config['tile_dir'], round_files, nb.basic_info.n_tiles)
    
    nbp.tile = tile_names.tolist()  # npy tile file paths list [n_tiles x n_rounds (x n_channels if 3D)]
    nb += nbp
