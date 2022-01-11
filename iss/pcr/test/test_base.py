import unittest
from ..base import get_transform, get_average_transform
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


class TestGetAverageTransform(unittest.TestCase):
    """
    Check whether getting average transform from all good transforms is the same as MATLAB.
    test files created using python_testing/PCR/get_average_transform.m script.

    test files contain:
    file: iss object used to create test
    D:  [4 x 3 x n_tiles x n_rounds x n_channels]
        D[:, :, t, r, c] is the affine transform for tile t from the reference image to round r, channel c.
    nMatches:  [n_tiles x n_channels x n_rounds] number of matches found by point cloud registration
    matches_thresh:  [n_tiles x n_channels x n_rounds], nMatches much exceed this to be a good transform.
    scale_thresh: [3,] if scaling to color channel differs from median by over this then it is a bad transform.
    shift_thresh: [3,] if shift to tile/round differs from median by over this then it is a bad transform.
    D_average:  [4 x 3 x n_tiles x n_rounds x n_channels]
        D_average[:, :, t, r, c] is the median affine transform found by MATLAB
    A: [n_channels x 3], median scaling found by MATLAB
    PcFailed: [n_tiles x n_channels x n_rounds], tiles/channels/rounds for which transform was bad.

    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/get_average_transform/')
    tol = 0

    def test_get_transform(self):
        test_files = [s for s in os.listdir(self.folder) if "test" in s and "fail" not in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(self.folder, file_name)
            transforms, n_matches, matches_thresh, scale_thresh, shift_thresh, av_transforms_matlab, av_scaling_matlab,\
                failed_matlab = \
                matlab.load_array(test_file, ['D', 'nMatches', 'matches_thresh', 'scale_thresh', 'shift_thresh',
                                              'D_average', 'A', 'PcFailed'])
            n_matches = np.moveaxis(n_matches, 1, 2)  # change to t,r,c from MATLAB t,c,r
            matches_thresh = np.moveaxis(matches_thresh, 1, 2)  # change to t,r,c from MATLAB t,c,r
            failed_matlab = np.moveaxis(failed_matlab, 1, 2)  # change to t,r,c from MATLAB t,c,r
            av_transforms_python, av_scaling_python, av_shifts, failed_python, failed_non_matches = \
                get_average_transform(transforms, n_matches, matches_thresh, scale_thresh, shift_thresh)
            diff_1 = av_transforms_python - av_transforms_matlab
            diff_2 = av_scaling_python - av_scaling_matlab
            diff_3 = failed_python.astype(int) - failed_matlab
            self.assertTrue(np.abs(diff_1).max() <= self.tol)
            self.assertTrue(np.abs(diff_2).max() <= self.tol)
            self.assertTrue(np.abs(diff_3).max() <= self.tol)

    @unittest.expectedFailure
    def test_get_transform_fail(self):
        # should hit error because not enough good shifts to compute median
        test_files = [s for s in os.listdir(self.folder) if "test" in s and "fail" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(self.folder, file_name)
            transforms, n_matches, matches_thresh, scale_thresh, shift_thresh, av_transforms_matlab, av_scaling_matlab,\
                failed_matlab = \
                matlab.load_array(test_file, ['D', 'nMatches', 'matches_thresh', 'scale_thresh', 'shift_thresh',
                                              'D_average', 'A', 'PcFailed'])
            n_matches = np.moveaxis(n_matches, 1, 2)  # change to t,r,c from MATLAB t,c,r
            matches_thresh = np.moveaxis(matches_thresh, 1, 2)  # change to t,r,c from MATLAB t,c,r
            failed_matlab = np.moveaxis(failed_matlab, 1, 2)  # change to t,r,c from MATLAB t,c,r
            av_transforms_python, av_scaling_python, av_shifts, failed_python, failed_non_matches = \
                get_average_transform(transforms, n_matches, matches_thresh, scale_thresh, shift_thresh)


if __name__ == '__main__':
    unittest.main()
