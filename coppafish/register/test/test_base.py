import unittest
from ..base import get_transform, get_average_transform, icp
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
                get_transform(yxz_base, transform_old, yxz_target, dist_thresh, reg_constant_scale=reg_constant_rot,
                              reg_constant_shift=reg_constant_shift, reg_transform=reg_transform)
            diff_1 = transform_python - transform_matlab
            diff_2 = (neighbour_python+1) - neighbour_matlab
            # Only consider neighbours where dist < dist_thresh.
            diff_2 = diff_2[neighbour_python != yxz_target.shape[0]]
            diff_3 = n_matches_python - n_matches_matlab
            diff_4 = error_python - error_matlab
            self.assertTrue(np.abs(diff_1).max() <= self.tol_transform)
            self.assertTrue(np.sum(diff_2 != 0) <= self.tol_neighb)
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
            transforms = np.moveaxis(transforms, [0, 1], [-2, -1])  # put t,r,c indices before dim for Python
            av_transforms_matlab = np.moveaxis(av_transforms_matlab, [0, 1], [-2, -1])
            n_matches = np.moveaxis(n_matches, 1, 2)  # change to t,r,c from MATLAB t,c,r
            matches_thresh = np.moveaxis(matches_thresh, 1, 2)  # change to t,r,c from MATLAB t,c,r
            failed_matlab = np.moveaxis(failed_matlab, 1, 2)  # change to t,r,c from MATLAB t,c,r
            scale_thresh = scale_thresh[0]
            shift_thresh = shift_thresh[0]
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
            transforms = np.moveaxis(transforms, [0, 1], [-2, -1])  # put t,r,c indices before dim for Python
            av_transforms_matlab = np.moveaxis(av_transforms_matlab, [0, 1], [-2, -1])
            n_matches = np.moveaxis(n_matches, 1, 2)  # change to t,r,c from MATLAB t,c,r
            matches_thresh = np.moveaxis(matches_thresh, 1, 2)  # change to t,r,c from MATLAB t,c,r
            failed_matlab = np.moveaxis(failed_matlab, 1, 2)  # change to t,r,c from MATLAB t,c,r
            scale_thresh = scale_thresh[0]
            shift_thresh = shift_thresh[0]
            av_transforms_python, av_scaling_python, av_shifts, failed_python, failed_non_matches = \
                get_average_transform(transforms, n_matches, matches_thresh, scale_thresh, shift_thresh)


class TestIterate(unittest.TestCase):
    """
    Check whether transforms found by point cloud registration over all iterations is the same as MATLAB.
    test files created using python_testing/PCR/iterate.m script.

    test files contain:
    file: iss object used to create test
    use_tiles, use_channels, use_rounds: tiles, channels, rounds used to run PCR.
    yxz_base:  [n_tiles] yxz_base[t] is the centered coordinates of anchor round of tile t.
    yxz_target: [n_tiles x n_channels x n_rounds]
        yxz_target[t,c,r] are the centered coordinates of tile t, channel c, round r.
    D0:  [4 x 3 x n_tiles x n_rounds x n_channels]
        affine transforms that are the starting transforms for the PCR algorithm.
    n_iter: how many iterations the PCR was run with
    dist_thresh: if spots closer than dist_thresh then they are used to compute the new transform.
    matches_thresh:  [n_tiles x n_channels x n_rounds], nMatches must exceed this to be a good transform.
    scale_thresh: [3,] if scaling to color channel differs from median by over this then it is a bad transform.
    shift_thresh: [3,] if shift to tile/round differs from median by over this then it is a bad transform.
    reg_constant_rot: constant used for scaling and rotation when doing regularised least squares.
    reg_constant_shift: constant used for shift when doing regularised least squares.
    reg_transform: [4x3] affine transform which we want final transform to be near when doing regularised least squares.
    D:  [4 x 3 x n_tiles x n_rounds x n_channels]
        affine transforms that were found by MATLAB PCR algorithm.
    A: [n_channels x 3], median scaling found by MATLAB
    n_matches:  [n_tiles x n_channels x n_rounds] number of matches found by MATLAB
    error: [n_tiles x n_channels x n_rounds] mean distance between matches found by MATLAB
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/iterate/')
    tol1 = 0.05
    tol_matches = 5
    tol2 = 5e-4
    tol_fract = 0.08

    def test_get_transform(self):
        test_files = [s for s in os.listdir(self.folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(self.folder, file_name)
            # to load big cell array as numpy object array, need use load_v_less_7_3
            yxz_base, yxz_target, transforms_start, n_iter, dist_thresh, matches_thresh, scale_dev_thresh,\
                shift_dev_thresh, reg_constant_rot, reg_constant_shift, transforms_matlab, av_scaling_matlab,\
                n_matches_matlab, error_matlab, failed_matlab = \
                matlab.load_v_less_7_3(test_file, ['yxz_base', 'yxz_target', 'D0', 'n_iter', 'dist_thresh',
                                                   'matches_thresh', 'scale_dev_thresh', 'shift_dev_thresh',
                                                   'reg_constant_rot', 'reg_constant_shift', 'D', 'A',
                                                   'n_matches', 'error', 'failed'])
            n_iter = int(n_iter)
            dist_thresh = float(dist_thresh)
            reg_constant_rot = float(reg_constant_rot)
            reg_constant_shift = float(reg_constant_shift)
            scale_dev_thresh = scale_dev_thresh[0]
            shift_dev_thresh = shift_dev_thresh[0]
            yxz_base = yxz_base.squeeze()
            # put t,r,c indices before dim for Python transforms
            transforms_start = np.moveaxis(transforms_start, [0, 1], [-2, -1])
            transforms_matlab = np.moveaxis(transforms_matlab, [0, 1], [-2, -1])
            n_matches_matlab = np.moveaxis(n_matches_matlab, 1, 2)  # change to t,r,c from MATLAB t,c,r
            matches_thresh = np.moveaxis(matches_thresh, 1, 2)  # change to t,r,c from MATLAB t,c,r
            error_matlab = np.moveaxis(error_matlab, 1, 2)  # change to t,r,c from MATLAB t,c,r
            failed_matlab = np.moveaxis(failed_matlab, 1, 2)  # change to t,r,c from MATLAB t,c,r
            yxz_target = np.moveaxis(yxz_target, 1, 2)  # change to t,r,c from MATLAB t,c,r
            transforms_python, debug_python = icp(yxz_base, yxz_target, transforms_start, n_iter, dist_thresh,
                                                  matches_thresh, scale_dev_thresh, shift_dev_thresh,
                                                  reg_constant_rot, reg_constant_shift)
            diff_1 = transforms_python - transforms_matlab
            diff_2 = debug_python['n_matches'] - n_matches_matlab
            diff_3 = debug_python['error'] - error_matlab
            diff_4 = debug_python['failed'].astype(int) - failed_matlab
            diff_5 = debug_python['av_scaling'] - av_scaling_matlab
            self.assertTrue(np.abs(diff_1).max() <= self.tol1 and
                            np.sum(np.abs(diff_1) > self.tol2)/np.prod(np.shape(diff_1)) < self.tol_fract)
            self.assertTrue(np.abs(diff_2).max() <= self.tol_matches and
                            np.sum(np.abs(diff_2) > self.tol2)/np.prod(np.shape(diff_2)) < self.tol_fract)
            self.assertTrue(np.abs(diff_3).max() <= self.tol1 and
                            np.sum(np.abs(diff_3) > self.tol2)/np.prod(np.shape(diff_3)) < self.tol_fract)
            self.assertTrue(np.abs(diff_4).max() <= self.tol1 and
                            np.sum(np.abs(diff_4) > self.tol2)/np.prod(np.shape(diff_4)) < self.tol_fract)
            self.assertTrue(np.abs(diff_5).max() <= self.tol1 and
                            np.sum(np.abs(diff_5) > self.tol2)/np.prod(np.shape(diff_5)) < self.tol_fract)


if __name__ == '__main__':
    unittest.main()
