import unittest
import os
import numpy as np
from ...utils import matlab, errors
from ..base import color_normalisation


class TestColorNormalisation(unittest.TestCase):
    """
    Check whether scaled color_normalisation works the same as MATLAB.
    test files were made with python_testing/call_spots/color_normalisation.m script.

    test files contain:
    InputDirectory: directory of data used to create unit test data
    method: 'single' or 'separate' whether to have one normalisation value across all rounds or one for each round.
    ChannelNormValue1, ChannelNormValue2, ChannelNormValue3
    ChannelNormThresh1, ChannelNormThresh2, ChannelNormThresh3:
        After normalisation, the probability of a pixel with intensity greater than each ChannelNormValue must be less
        than the corresponding ChannelNormThresh.
    HistCounts: integer array [n_hist_values x n_channels x n_rounds]
        the number of pixels found for each intensity value in each round and channel
    HistValues: integer array [n_hist_values]
        all possible intensity values in tiff images.
    p: float array [n_channels x n_rounds], final normalisation found for each round and channel.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_color_normalisation(self):
        folder = os.path.join(self.folder, 'color_normalisation')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            method, norm_val_1, norm_thresh_1, norm_val_2, norm_thresh_2, norm_val_3, norm_thresh_3, hist_counts,\
                hist_values, p_matlab = \
                matlab.load_array(test_file, ['method', 'ChannelNormValue1', 'ChannelNormThresh1', 'ChannelNormValue2',
                                              'ChannelNormThresh2', 'ChannelNormValue3', 'ChannelNormThresh3',
                                              'HistCounts', 'HistValues', 'p'])
            if method.shape[1] == 8:
                method = 'separate'
            elif method.shape[1] == 6:
                method = 'single'
            else:
                method = 'blah'
            thresh_intensities = [float(norm_val_1), float(norm_val_2), float(norm_val_3)]
            thresh_probs = [float(norm_thresh_1), float(norm_thresh_2), float(norm_thresh_3)]
            hist_values = hist_values.flatten().astype(int)
            hist_counts = np.moveaxis(hist_counts, 1, 2).astype(int)  # change to r,c from MATLAB c,r
            p_matlab = np.moveaxis(p_matlab, 0, 1)  # change to r,c from MATLAB c,r

            p_python = color_normalisation(hist_values, hist_counts, thresh_intensities, thresh_probs, method)
            diff = p_python - p_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)
