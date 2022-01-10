import unittest
from ..base import get_transform
from ...utils import matlab, errors
import os
import numpy as np


class TestGetTransform(unittest.TestCase):
    """
    Check whether transform found by point cloud registration on a single iteration is the same as MATLAB.
    test files created using python_testing/PCR/get_transform.m script.

    test files contain:
    file: iss object used to create test
    t, b, r: tile, channel, round from which spots come from.
    yxz_base:  centered coordinates of anchor round of tile t.
    transform_old:  [4x3] affine transform that is the starting transform for the PCR algorithm.
    yxz_target:  centered coordinates of tile t, channel b, round r.
    dist_thresh: if spots closer than dist_thresh then they are used to compute the new transform.
    reg_constant_rot: constant used for scaling and rotation when doing regularised least squares.
    reg_constant_shift: constant used for shift when doing regularised least squares.
    reg_transform: [4x3] affine transform which we want final transform to be near when doing regularised least squares.
    transform: [4x3] affine transform found by MATLAB
    neighbour: neighbour[i] is index of coordinate in yxz_target to which transformation of yxz_base[i] is closest.
    n_matches: number of neighbours which have distance between them of less than dist_thresh.
    error: average distance between neighbours below dist_thresh.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/get_transform/')
    tol_transform = 1e-10
    tol_neighb = 100

    def test_get_transform(self):
        test_files = [s for s in os.listdir(self.folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(self.folder, file_name)
            yxz_base, transform_old, yxz_target, dist_thresh, reg_constant_rot, reg_constant_shift, reg_transform,\
                transform_matlab, neighbour_matlab, n_matches_matlab, error_matlab = \
                matlab.load_array(test_file, ['yxz_base', 'transform_old', 'yxz_target', 'dist_thresh',
                                              'reg_constant_rot', 'reg_constant_shift', 'reg_transform', 'transform',
                                              'neighbour', 'n_matches', 'error'])
            n_matches_matlab = int(n_matches_matlab)
            dist_thresh = float(dist_thresh)
            neighbour_matlab = neighbour_matlab.flatten().astype(int)
            error_matlab = float(error_matlab)
            if np.abs(reg_transform).max() == 0:
                reg_transform = None
            transform_python, neighbour_python, n_matches_python, error_python = \
                get_transform(yxz_base, transform_old, yxz_target, dist_thresh, reg_constant_rot=reg_constant_rot,
                              reg_constant_shift=reg_constant_shift, reg_transform=reg_transform)
            diff_1 = transform_python - transform_matlab
            diff_2 = (neighbour_python+1) - neighbour_matlab
            diff_3 = n_matches_python - n_matches_matlab
            diff_4 = error_python - error_matlab
            self.assertTrue(np.abs(diff_1).max() <= self.tol_transform)
            self.assertTrue(sum(diff_2 != 0) <= self.tol_neighb)
            self.assertTrue(np.abs(diff_3) <= self.tol_transform)
            self.assertTrue(np.abs(diff_4) <= self.tol_neighb)


if __name__ == '__main__':
    unittest.main()
