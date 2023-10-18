from typing import Optional
import numpy as np

from ..setup import NotebookPage


def get_shifts_to_search(config: dict, nbp_basic: NotebookPage, nbp_debug: Optional[NotebookPage] = None) -> dict:
    """
    Using information in config dictionary to get range of shifts to search over when finding overlap between
    overlapping tiles in north and east directions.

    Args:
        config: 'stitch' section of .ini document.
        nbp_basic: `basic_info` notebook page
        nbp_debug: `stitch` notebook page where debugging information for stitching is kept.
            If provided, north_start_shift_search and east_start_shift_search variables added to page.

    Returns:
        `shifts[j][i]` contains the shifts to search over for overlap in the `j` direction for coordinate `i` where:
            `j = 'north', 'east'`.
            `i = 'y', 'x', 'z'`.
    """
    expected_shift_north = np.array([-(1 - config['expected_overlap']) * nbp_basic.tile_sz, 0, 0]).astype(int)
    auto_shift_north_extent = np.array(config['auto_n_shifts']) * np.array(config['shift_step'])
    expected_shift_east = expected_shift_north[[1, 0, 2]]
    auto_shift_east_extent = auto_shift_north_extent[[1, 0, 2]]
    if config['shift_north_min'] is None:
        config['shift_north_min'] = list(expected_shift_north - auto_shift_north_extent)
    if config['shift_north_max'] is None:
        config['shift_north_max'] = list(expected_shift_north + auto_shift_north_extent)
    if config['shift_east_min'] is None:
        config['shift_east_min'] = list(expected_shift_east - auto_shift_east_extent)
    if config['shift_east_max'] is None:
        config['shift_east_max'] = list(expected_shift_east + auto_shift_east_extent)
    directions = ['north', 'east']
    coords = ['y', 'x', 'z']
    shifts = {'north': {}, 'east': {}}
    for j in directions:
        if nbp_debug is not None:
            nbp_debug.__setattr__(j + '_' + 'start_shift_search', np.zeros((3, 3), dtype=int))
        for i in range(len(coords)):
            shifts[j][coords[i]] = np.arange(config['shift_' + j + '_min'][i],
                                             config['shift_' + j + '_max'][i] +
                                             config['shift_step'][i] / 2, config['shift_step'][i]).astype(int)
            if nbp_debug is not None:
                nbp_debug.__getattribute__(j + '_' + 'start_shift_search')[i, :] = [config['shift_' + j + '_min'][i],
                                                                                    config['shift_' + j + '_max'][i],
                                                                                    config['shift_step'][i]]
    if not nbp_basic.is_3d:
        shifts['north']['z'] = np.array([0], dtype=int)
        shifts['east']['z'] = np.array([0], dtype=int)
        if nbp_debug is not None:
            for j in directions:
                nbp_debug.__getattribute__(j + '_' + 'start_shift_search')[2, :2] = 0
    return shifts
