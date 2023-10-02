from coppafish.utils.errors import check_shape, check_color_nan
from coppafish import NotebookPage

import numpy as np


def test_check_shape():
    shape = (5, 10, 4, 2)
    wrong_shape_1 = (4, 10, 4, 2)
    wrong_shape_2 = (5, 10, 1, 2)
    wrong_shape_3 = (5, 10, 4)
    wrong_shape_4 = (5, 10, 4, 2, 20)
    array = np.empty(shape)
    assert check_shape(array, shape)
    assert not check_shape(array, wrong_shape_1)
    assert not check_shape(array, wrong_shape_2)
    assert not check_shape(array, wrong_shape_3)
    assert not check_shape(array, wrong_shape_4)
    assert check_shape(array, list(shape))
    assert not check_shape(array, list(wrong_shape_1))
    assert not check_shape(array, list(wrong_shape_2))
    assert not check_shape(array, list(wrong_shape_3))
    assert not check_shape(array, list(wrong_shape_4))
    assert check_shape(array, np.array(shape))
    assert not check_shape(array, np.array(wrong_shape_1))
    assert not check_shape(array, np.array(wrong_shape_2))
    assert not check_shape(array, np.array(wrong_shape_3))
    assert not check_shape(array, np.array(wrong_shape_4))


def test_check_color_nan():
    n_codes = 2
    n_rounds = 6
    n_channels = 8
    shape = (n_codes, n_rounds, n_channels)
    nbp_basic = NotebookPage('nbp_basic', 
                             {'tile_pixel_value_shift':10, 'use_rounds': [0,1,2,4], 'use_channels': [1,6,7], 
                              'n_rounds':n_rounds, 'n_channels':n_channels})
    rng = np.random.RandomState(38)
    # Place the correct invalid value expected by the function in the unused rounds/channels
    array = np.full(shape, fill_value=np.nan, dtype=float)
    for s in range(n_codes):
        for r in range(n_rounds):
            for c in range(n_channels):
                if not r in nbp_basic.use_rounds:
                    continue
                if not c in nbp_basic.use_channels:
                    continue
                # Set co,r,c to a non invalid value
                array[s,r,c] = rng.rand()
    check_color_nan(array, nbp_basic)
    del array
    array = np.full(shape, fill_value=-nbp_basic.tile_pixel_value_shift, dtype=int)
    for s in range(n_codes):
        for r in range(n_rounds):
            for c in range(n_channels):
                if not r in nbp_basic.use_rounds:
                    continue
                if not c in nbp_basic.use_channels:
                    continue
                # Set co,r,c to a non invalid value
                array[s,r,c] = rng.randint(0, 100, dtype=int)
    check_color_nan(array, nbp_basic)

