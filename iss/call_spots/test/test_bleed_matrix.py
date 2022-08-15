import unittest
import os
import numpy as np
from ...utils import matlab, errors
from ..bleed_matrix import scaled_k_means, get_bleed_matrix, get_dye_channel_intensity_guess


class TestScaledKMeans(unittest.TestCase):
    """
    Check whether scaled k means works the same as MATLAB.

    test files contain:
    x: data set of vectors to build clusters from [n_points x n_dims]
    v0: starting point of mean cluster vectors [n_clusters x n_dims]
    ScoreThresh: float between 0 and 1, points in x with dot product to a cluster mean greater than this
                 contribute to new estimate of mean vector.
    MinClusterSize: integer, if less than this many points assigned to a cluster, that cluster mean will be set to 0.
    ConvergenceCriterion: integer, when less than this many points assigned to a different cluster than
                          previous iteration, algorithm will terminate.
    MaxIter: integer, maximum number of iterations performed.
    k: integer array [n_points], cluster that each point assigned to (0 means falls below ScoreThresh and not assigned)
    v: final mean cluster vectors [n_clusters x n_dims]
    s2: float array [n_clusters], first eigenvalue of the outer product matrix for each cluster
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_scaled_k_means(self):
        folder = os.path.join(self.folder, 'scaled_k_means')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            x, v0, score_thresh, min_cluster_size, convergence_criterion, max_iter, k_matlab, v_matlab, s2_matlab = \
                matlab.load_array(test_file, ['x', 'v0', 'ScoreThresh', 'MinClusterSize', 'ConvergenceCriterion',
                                              'MaxIter', 'k', 'v', 's2'])
            score_thresh = float(score_thresh)
            max_iter = int(max_iter)
            min_cluster_size = int(min_cluster_size)
            k_matlab = k_matlab.flatten() - 1
            s2_matlab = s2_matlab.flatten()
            v_python, s2_python, k_python, k_score = scaled_k_means(x, v0, score_thresh, min_cluster_size, max_iter)[:4]
            diff_1 = v_python - v_matlab
            diff_2 = k_python - k_matlab
            diff_3 = s2_python - s2_matlab
            self.assertTrue(np.abs(diff_1).max() <= self.tol)
            self.assertTrue(np.abs(diff_2).max() <= self.tol)
            self.assertTrue(np.abs(diff_3).max() <= self.tol)


class TestGetBleedMatrix(unittest.TestCase):
    """
    Check whether bleed matrix is obtained is the same as MATLAB.
    test files were made with python_testing/call_spots/get_bleed_matrix.m script.

    test files contain:
    InputDirectory: directory of data used to create unit test data
    SpotColors: spot colors to find bleed matrix from [n_spots x n_channels x n_rounds]
    InitialBleedMatrix: starting point of intensity of each dye in each channel [n_dyes x n_channels]
    ScoreThresh: float between 0 and 1, points in x with dot product to a cluster mean greater than this
                 contribute to new estimate of mean vector.
    method: 'single' or 'separate' whether to have one bleed matrix across all rounds or one for each.
    BleedMatrix: [n_rounds x n_channels x n_dyes] bleed matrix for each round.
    MinClusterSize: integer, if less than this many points assigned to a cluster in scaled K means,
                    that cluster mean will be set to 0.
    MaxIter: integer, maximum number of iterations performed in scaled K means.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_get_bleed_matrix(self):
        folder = os.path.join(self.folder, 'get_bleed_matrix')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            spot_colors, dye_channel_matrix, score_thresh, method, bleed_matrix_matlab, n_iter, min_cluster_size = \
                matlab.load_array(test_file, ['SpotColors', 'InitialBleedMatrix', 'ScoreThresh', 'method',
                                              'BleedMatrix', 'MaxIter', 'MinClusterSize'])
            spot_colors = np.moveaxis(spot_colors, 1, 2)  # change to r,c from MATLAB c,r
            score_thresh = float(score_thresh)
            n_iter = int(n_iter)
            min_cluster_size = int(min_cluster_size)
            if method.shape[1] == 8:
                method = 'separate'
            elif method.shape[1] == 6:
                method = 'single'
            else:
                method = 'blah'
            initial_bleed_matrix = np.expand_dims(dye_channel_matrix.transpose(), 0)
            n_rounds = bleed_matrix_matlab.shape[0]
            initial_bleed_matrix = np.tile(initial_bleed_matrix, (n_rounds, 1, 1))
            # all tests were run with score_thresh_anneal = False
            bleed_matrix_python = get_bleed_matrix(spot_colors, initial_bleed_matrix, method, score_thresh,
                                                   min_cluster_size, n_iter, False)
            diff = bleed_matrix_python - bleed_matrix_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)


class TestGetDyeChannelIntensityGuess(unittest.TestCase):
    """
    Check whether bleed matrix is obtained is the same as MATLAB.
    test files were made with python_testing/call_spots/get_bleed_matrix.m script.

    test files contain:
    file_name: path of csv file containing raw dye, camera, laser intensity data
    dye_names: cell array [n_dyes] giving name of dyes used
    cameras: integer array [n_channels] giving waelength in nm of camera used in each channel
    lasers: integer array [n_channels] giving wavelength in nm of laser used in each channel
    output: float array [n_dyes x n_channels] giving approximate intensity of each dye in each channel
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    csv_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'setup', 'dye_camera_laser_raw_intensity.csv')
    tol = 1e-10

    def test_get_dye_channel_intensity_guess(self):
        folder = os.path.join(self.folder, 'get_dye_channel_intensity_guess')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            csv_file_name, dye_names, cameras, lasers, output_matlab = \
                matlab.load_v_less_7_3(test_file, ['file_name', 'dye_names', 'cameras', 'lasers', 'output'])
            dye_names = [dye_names[i][0][0] for i in range(dye_names.shape[0])]
            cameras = cameras.astype(int).flatten()
            lasers = lasers.astype(int).flatten()
            output_python = get_dye_channel_intensity_guess(self.csv_file, dye_names, cameras, lasers)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)
