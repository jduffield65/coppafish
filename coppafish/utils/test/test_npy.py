import os
import numpy as np

from coppafish import NotebookPage
from coppafish.utils import npy


def test_npy_save_load_tile():
    directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unit_test_dir')

    if not os.path.isdir(directory):
        os.mkdir(directory)
    rng = np.random.RandomState(0)
    array_1 = rng.randint(2**16, size=(3,3,4), dtype=np.int32)
    array_2 = rng.randint(2**16, size=(2,4,4), dtype=np.int32)
    nbp_file_3d = NotebookPage('file_names', {
        'tile': [[[os.path.join(directory, 'array.npy')]]]
    })
    nbp_file_2d = NotebookPage('file_names', {
        'tile': [[os.path.join(directory, 'array.npy')]]
    })
    nbp_basic_3d = NotebookPage('basic_info', {
        'is_3d': True,
        'anchor_round': 100,
        'dapi_channel': 5,
        'tile_sz': 3,
        'use_z': [0,1,2,3],
        'tile_pixel_value_shift': 0,
    })
    nbp_basic_2d = NotebookPage('basic_info', {
        'is_3d': False,
        'anchor_round': 100,
        'dapi_channel': 5,
        'tile_sz': 4,
        'use_z': [0,1,2,3],
        'tile_pixel_value_shift': 0,
        'use_channels': [0,1],
        'n_channels': 2,
    })
    # 3d
    npy.save_tile(nbp_file_3d, nbp_basic_3d, array_1, 0, 0, 0)
    output = npy.load_tile(nbp_file_3d, nbp_basic_3d, 0, 0, 0, None)
    assert np.allclose(array_1, output), 'Loaded in tile does not have the same values as starting tile'
    npy.save_tile(nbp_file_3d, nbp_basic_3d, array_1, 0, 0, 0, num_rotations=2)
    output = npy.load_tile(nbp_file_3d, nbp_basic_3d, 0, 0, 0, None)
    assert np.allclose(np.rot90(array_1, 2, axes=(0, 1)), output), 'Expected a rotated tile to be loaded in'
    yxz = [None, None, 1]
    npy.save_tile(nbp_file_3d, nbp_basic_3d, array_1, 0, 0, 0)
    output = npy.load_tile(nbp_file_3d, nbp_basic_3d, 0, 0, 0, yxz=yxz)
    assert np.allclose(array_1[:,:,1], output), 'Expected a subvolume to be loaded in'
    # TODO: Make 2d work, potentially errors in the script itself
    # npy.save_tile(nbp_file_2d, nbp_basic_2d, array_2, 0, 0, 0, num_rotations=1, suffix='suffix_array_2')
    # output = npy.load_tile(nbp_file_2d, nbp_basic_2d, 0, 0, 0, suffix='suffix_array_2')


# TODO: save_stitched and get_npy_tile_ind unit tests
