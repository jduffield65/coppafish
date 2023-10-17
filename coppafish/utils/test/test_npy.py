from coppafish.utils.npy import load_tile, save_tile, save_stitched, get_npy_tile_ind

import numpy as np


def test_npy_save_load_tile():
    rng = np.random.RandomState(0)
    array = rng.randint(2**31, size=(2,3,4), dtype=np.int32)
    # save_tile()
