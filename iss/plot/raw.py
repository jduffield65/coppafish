import napari
from .. import setup
from ..utils import raw
from ..pipeline.basic_info import set_basic_info
import numpy as np
import numbers
from typing import Union, Optional, List
import os
from tqdm import tqdm


def view_raw(config_file: str, rounds: Union[int, List[int]], tiles: Union[int, List[int]],
             channels: Optional[Union[int, List[int]]] = None):
    """
    Function to view raw data in napari

    Args:
        config_file: path to config file for experiment
        rounds: rounds to view
        tiles: npy (as opposed to nd2 fov) tile indices to view
        channels: channels to view
    """
    config = setup.get_config(config_file)
    if not config['file_names']['notebook_name'].endswith('.npz'):
        # add .npz suffix if not in name
        config['file_names']['notebook_name'] = config['file_names']['notebook_name'] + '.npz'
    nb_path = os.path.join(config['file_names']['output_dir'], config['file_names']['notebook_name'])
    if os.path.isfile(nb_path):
        nb = setup.Notebook(nb_path, config_file)
    else:
        nb = setup.Notebook('empty', config_file)
    if not nb.has_page("basic_info"):
        nb._no_save_pages['basic_info'] = {}  # don't save if add basic_info page
        nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        nb += nbp_basic

    if isinstance(rounds, numbers.Number):
        rounds = [rounds]
    if isinstance(tiles, numbers.Number):
        tiles = [tiles]
    if channels is None:
        channels = np.arange(nb.basic_info.n_channels)
    if isinstance(channels, numbers.Number):
        channels = [channels]

    viewer = napari.Viewer()
    n_images = len(rounds) * len(tiles) * len(channels)
    with tqdm(total=n_images) as pbar:
        pbar.set_description(f'Loading in raw data')
        for r in rounds:
            round_dask_array = raw.load(nb.file_names, nb.basic_info, r=r)
            # TODO: Can get rid of these two for loops, when round_dask_array is always a dask array.
            #  At the moment though, is not dask array when using nd2_reader (On Mac M1).
            for t in tiles:
                all_channel_image = np.zeros((nb.basic_info.n_channels, nb.basic_info.nz, nb.basic_info.tile_sz,
                                              nb.basic_info.tile_sz), dtype=np.uint16)
                for c in channels:
                    pbar.set_postfix({'round': r, 'tile': t, 'channel': c})
                    all_channel_image[c] = np.moveaxis(raw.load(nb.file_names, nb.basic_info, round_dask_array, r, t, c,
                                                                nb.basic_info.use_z), -1, 0)
                    pbar.update(1)
                viewer.add_image(all_channel_image, name=f"Round {r}, Tile {t} Raw Data")
    viewer.dims.axis_labels = ['channel', 'z', 'y', 'x']
    napari.run()
