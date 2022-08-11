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

    # remove file extension from round and anchor file names if it is present
    if config['round'] is None:
        if config['anchor'] is None:
            raise ValueError(f'Neither imaging rounds nor anchor_round provided')
        config['round'] = []  # Sometimes the case where just want to run the anchor round.
    config['round'] = [r.replace(config['raw_extension'], '') for r in config['round']]
    nbp.round = config['round']

    if config['anchor'] is not None:
        config['anchor'] = config['anchor'].replace(config['raw_extension'], '')
    nbp.anchor = config['anchor']
    nbp.raw_extension = config['raw_extension']
    nbp.raw_metadata = config['raw_metadata']

    if config['dye_camera_laser'] is None:
        # Default information is project
        config['dye_camera_laser'] = os.path.join(os.path.dirname(__file__), 'dye_camera_laser_raw_intensity.csv')
    nbp.dye_camera_laser = config['dye_camera_laser']
    config['code_book'] = config['code_book'].replace('.txt', '')
    nbp.code_book = config['code_book'] + '.txt'

    # where to save scale and scale_anchor values used in extract step.
    config['scale'] = config['scale'].replace('.txt', '')
    nbp.scale = os.path.join(config['tile_dir'], config['scale'] + '.txt')

    # where to save psf, indicating average spot shape in raw image. Only ever needed in 3D.
    if nb.basic_info.is_3d:
        config['psf'] = config['psf'].replace('.npy', '')
        nbp.psf = os.path.join(config['output_dir'], config['psf'] + '.npy')
    else:
        nbp.psf = None

    # where to save omp_spot_shape, indicating average spot shape in omp coefficient sign images.
    config['omp_spot_shape'] = config['omp_spot_shape'].replace('.npy', '')
    omp_spot_shape_file = os.path.join(config['output_dir'], config['omp_spot_shape'] + '.npy')
    nbp.omp_spot_shape = omp_spot_shape_file

    # Add files so save omp results after each tile as security if hit any bugs
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

    if nb.basic_info.is_3d:
        tile_names = get_tile_file_names(config['tile_dir'], round_files, nb.basic_info.n_tiles,
                                         nb.basic_info.n_channels)
    else:
        tile_names = get_tile_file_names(config['tile_dir'], round_files, nb.basic_info.n_tiles)

    nbp.tile = tile_names.tolist()  # npy tile file paths list [n_tiles x n_rounds (x n_channels if 3D)]
    nb += nbp
