import unittest
import os
import numpy as np
from ...utils import matlab, errors
from ..base import fitting_standard_deviation, fit_coefs


class TestFittingStandardDeviation(unittest.TestCase):
    """
    Check whether fitting_standard_deviation works the same as MATLAB function:
    iss-Josh/@iss_OMP/get_variance_weight.m with beta a single number.

    test files contain:
    bled_codes: float array [nGenes x nChannels x nRounds]
        Expected code for each gene
    coef: float array [nSpots x nCodes]
        coef of each code for each spot.
    alpha: float
        constant, by how much to increase variance as genes added.
    beta: float
        standard deviation in image with no genes added.
    sigma: float array [nSpots x nChannels x nRounds]
        standard deviation for each spot in each round/channel.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_fitting_standard_deviation(self):
        folder = os.path.join(self.folder, 'fitting_standard_deviation')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            bled_codes, coef, alpha, beta, output_matlab = \
                matlab.load_array(test_file, ['bled_codes', 'coef', 'alpha', 'beta', 'sigma'])
            bled_codes = np.moveaxis(bled_codes, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            output_matlab = np.moveaxis(output_matlab, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            output_python = fitting_standard_deviation(bled_codes, coef, float(alpha), float(beta))
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)


class TestFitCoefs(unittest.TestCase):
    """
    Check whether fit_coefs works the same as MATLAB function:
    iss-Josh/@iss_OMP/get_spot_residual.

    test files contain:
    bled_codes: float array [nGenes x nChannels x nRounds]
        Expected code for each gene
    spot_colors: float array [nSpots x nChannels x nRounds]
        Code of each spot.
    A_omega: float array [(nChannels x nRounds) x nGenes]
        flattened version of bled_codes, used as input for MATLAB function.
        Related by: bled_codes = permute(reshape(A_omega,[nChannels,nRounds,nGenes]), [3,1,2]);
    b: float array [(nChannels x nRounds) x nSpots]
        flattened version of spot_colors, used as input for MATLAB function.
        Related by: spot_colors = permute(reshape(b,[nChannels,nRounds,nSpots]), [3,1,2]);
    r: float array [nSpots x nChannels x nRounds]
        spot_colors after genes removed i.e. spot color residual.
    x_ls: float array [nSpots x nGenes]
        Coefficient of each gene for each spot.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_fit_coefs(self):
        folder = os.path.join(self.folder, 'fit_coefs')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            bled_codes, spot_colors, residual_matlab, coefs_matlab = \
                matlab.load_array(test_file, ['bled_codes', 'spot_colors', 'r', 'x_ls'])
            bled_codes = np.moveaxis(bled_codes, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            spot_colors = np.moveaxis(spot_colors, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            residual_matlab = np.moveaxis(residual_matlab, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            coefs_matlab = coefs_matlab.transpose()  # change to g,s from MATLAB s,g.
            n_genes = bled_codes.shape[0]
            n_spots, n_rounds, n_channels = spot_colors.shape
            residual_python, coefs_python = fit_coefs(bled_codes.reshape(n_genes, -1).transpose(),
                                                      spot_colors.reshape(n_spots, -1).transpose())
            residual_python = residual_python.transpose().reshape(n_spots, n_rounds, n_channels)
            diff1 = residual_python - residual_matlab
            diff2 = coefs_python - coefs_matlab
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)
