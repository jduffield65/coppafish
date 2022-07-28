import napari
from .. import setup
from ..utils import raw
from ..pipeline.basic_info import set_basic_info
import numpy as np
import numbers
from typing import Union, Optional, List
import os
from tqdm import tqdm


def add_basic_info_no_save(nb: setup.Notebook):
    """
    This adds the `basic_info` page to the notebook without saving the notebook.

    Args:
        nb: Notebook with no `basic_info` page.

    """
    if not nb.has_page("basic_info"):
        nb._no_save_pages['basic_info'] = {}  # don't save if add basic_info page
        config = nb.get_config()
        nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        nb += nbp_basic



def view_raw(config_file: str, rounds: Union[int, List[int]], tiles: Union[int, List[int]],
             channels: Optional[Union[int, List[int]]] = None):
    """
    Function to view raw data in napari

    Args:
        config_file: path to config file for experiment
        rounds: rounds to view
        tiles: npy (as opposed to nd2 fov) tile indices to view.
            For an experiment where the tiles are arranged in a 4 x 3 (ny x nx) grid, tile indices are indicated as
            below:

            | 2  | 1  | 0  |

            | 5  | 4  | 3  |

            | 8  | 7  | 6  |

            | 11 | 10 | 9  |
        channels: Channels to view. If `None`, will load all channels.
            Channels not included here will just be shown as all zeros.
    """
    nb = setup.Notebook(config_file=config_file)
    if not nb.has_page("basic_info"):
        add_basic_info_no_save(nb)

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
