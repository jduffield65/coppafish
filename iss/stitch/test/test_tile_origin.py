import unittest
from ...utils import matlab, errors
from ..tile_origin import get_tile_origin
import os
import numpy as np


class TestTileOrigin(unittest.TestCase):
    """
    Check whether nd2 tile indices are the same as got with MATLAB
    Also check whether tiff first and last indices make sense.

    test files contain:
    VerticalPairs: nPairs x 2 array. VerticalPairs(:,1) is the tile index of the tile to the
        south of VerticalPairs(:,0). (In matlab code, axis 1 and 0 are flipped, but flipped them before saving to test).
    vShifts: nPairs x 3 (or 2) array. YXZ shift from VerticalPairs(:,0) to VerticalPairs(:,1)
    HorizontalPairs: nPairs x 2 array. HorizontalPairs(:,1) is the tile index of the tile to the
        west of HorizontalPairs(:,0)
    hShifts: nPairs x 3 (or 2) array. YXZ shift from HorizontalPairs(:,0) to HorizontalPairs(:,1)
    nTiles: number of tiles in data set.
    HomeTile: tile to fix coordinate to build coordianate system about.
    RefPos: nTiles x 3 (or 2) array. YXZ origin of each tile in global coordinate system.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/tile_origin/')
    tol = 0.1

    def test_get_tilepos(self):
        test_files = [s for s in os.listdir(self.folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(self.folder, file_name)
            v_pairs, v_shifts, h_pairs, h_shifts, n_tiles, home_tile, output_matlab = \
                matlab.load_array(test_file, ['VerticalPairs', 'vShifts', 'HorizontalPairs', 'hShifts',
                                              'nTiles', 'HomeTile', 'RefPos'])
            ndims = output_matlab.shape[1]
            if np.ndim(v_pairs) == 1:
                # if matlab array empty, saves weird with only one dimension
                v_pairs = np.zeros((0, 2), dtype=int)
                v_shifts = np.zeros((0, ndims), dtype=int)
            if np.ndim(h_pairs) == 1:
                # if matlab array empty, saves weird with only one dimension
                h_pairs = np.zeros((0, 2), dtype=int)
                h_shifts = np.zeros((0, ndims), dtype=int)
            if ndims == 2:
                # add 0 z-shift for all tiles if 2d
                h_shifts = np.pad(h_shifts, [(0, 0), (0, 1)])
                v_shifts = np.pad(v_shifts, [(0, 0), (0, 1)])
                output_matlab = np.pad(output_matlab, [(0, 0), (0, 1)], constant_values=1)
            # correct matlab indexing
            # include round because otherwise had shifts of -2 going to -1.
            v_shifts = np.round(v_shifts).astype(int)
            v_pairs = np.round((v_pairs - 1)).astype(int)
            h_shifts = np.round(h_shifts).astype(int)
            h_pairs = np.round((h_pairs - 1)).astype(int)
            n_tiles = np.round(n_tiles).astype(int).item()
            home_tile = np.round((home_tile - 1)).astype(int).item()
            output_matlab = output_matlab - 1
            output_python = get_tile_origin(v_pairs, v_shifts, h_pairs, h_shifts, n_tiles, home_tile)
            diff = output_python - output_matlab
            self.assertTrue(np.nanmax(np.abs(diff)) <= self.tol)
            # check there are tiles with y, x, z origin equal to 0.
            self.assertTrue(np.nanmin(output_python, axis=0).max() <= self.tol)