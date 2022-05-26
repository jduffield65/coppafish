import unittest
import os
import numpy as np
from ...utils import matlab, errors
from ..base import color_normalisation, dot_product_score, fit_background, get_gene_efficiency, \
    fit_background_jax_vectorised, get_spot_intensity, get_spot_intensity_vectorised
import jax.numpy as jnp


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


class TestDotProductScore(unittest.TestCase):
    """
    Check whether DotProductScore works the same as MATLAB function:
    iss-Josh/@iss_OMP/get_weight_dot_product.m with o.ompNormBledCodesWeightDotProduct=true

    test files contain:
    SpotColors: float array [nSpots x nChannels x nRounds]
        intensity of each spot in each channel, round.
    BledCodes: float array [nGenes x nChannels x nRounds]
        expected intensity of each gene in each channel, round.
    norm_shift: float
        shift to apply to normalisation of SpotColors i.e. weak Spots get lower score.
    weight: [] or float array [nSpots x nChannels x nRounds]
        weight to apply to each channel, round for each spot when computing dot product.
        Empty if no weight.
    SpotScores: float array [nSpots x nGenes]
        dot product score for each spot/gene contribution. Max value is approx 1.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_dot_product(self):
        folder = os.path.join(self.folder, 'dot_product_score')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            spot_colors, bled_codes, norm_shift, weight, output_matlab = \
                matlab.load_array(test_file, ['SpotColors', 'BledCodes', 'norm_shift', 'weight', 'SpotScores'])
            spot_colors = np.moveaxis(spot_colors, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            bled_codes = np.moveaxis(bled_codes, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            norm_shift = float(norm_shift)
            n_rounds, n_channels = spot_colors[0].shape
            if weight.max() == 0:
                # empty array in Matlab loaded as [0, 0].
                weight_squared = None
            else:
                weight_squared = np.moveaxis(weight, 1, 2).astype(float) ** 2 # change to r,c from MATLAB c,r
                weight_squared = weight_squared.reshape(-1, n_rounds * n_channels)
            output_python = dot_product_score(spot_colors.reshape(-1, n_rounds * n_channels),
                                              bled_codes.reshape(-1, n_rounds * n_channels), norm_shift, weight_squared)
            diff = output_python - output_matlab
            self.assertTrue(np.abs(diff).max() <= self.tol)


class TestFitBackground(unittest.TestCase):
    """
    Check whether fit_background works the same as MATLAB function:
    iss-Josh/@iss_DotProduct/fit_background.m with BackgroundWeightPower=1.

    test files contain:
    spot_colors: float array [nSpots x nChannels x nRounds]
        intensity of each spot in each channel, round.
    weight_shift: float
        shift to apply to background weighting to stop blow up to infinity.
    residual: float array [nSpots x nChannels x nRounds]
        spot_colors after background removed.
    x_ls: float array [nSpots x nChannels]
        Coefficient of each background vector fit.
        BackgroundVectors are just 1 in relavent channel so do not have L2 norm of 1.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-6

    def test_fit_background(self):
        folder = os.path.join(self.folder, 'fit_background')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            spot_colors, weight_shift, residual_matlab, coef_matlab = \
                matlab.load_array(test_file, ['spot_colors', 'weight_shift', 'residual', 'x_ls'])
            spot_colors = np.moveaxis(spot_colors, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            residual_matlab = np.moveaxis(residual_matlab, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            weight_shift = float(weight_shift)

            # matlab background vectors do not have L2 norm = 1 but all non-zero elements are 1 so easy to compensate
            n_rounds = spot_colors.shape[1]
            matlab_norm_factor = np.linalg.norm(np.ones(n_rounds))
            coef_matlab = coef_matlab * matlab_norm_factor

            residual_python, coef_python, background_vectors = fit_background(spot_colors, weight_shift)
            residual_jax, coef_jax, background_vectors_jax = fit_background_jax_vectorised(spot_colors, weight_shift)
            diff1 = residual_python - residual_matlab
            diff2 = coef_python - coef_matlab
            diff1_jax = np.asarray(residual_jax) - residual_python
            diff2_jax = np.asarray(coef_jax) - coef_python
            diff3_jax = np.asarray(background_vectors_jax) - background_vectors
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)
            self.assertTrue(np.abs(diff1_jax).max() <= self.tol)
            self.assertTrue(np.abs(diff2_jax).max() <= self.tol)
            self.assertTrue(np.abs(diff3_jax).max() <= self.tol)


class TestGetGeneEfficiency(unittest.TestCase):
    """
    Check whether gene_efficiencies function works the same as MATLAB function.
    Test data got using the function:
    python_testing/call_spots/gene_efficiencies.m

    test files contain:
    SpotColors: float array [nSpots x nChannels x nRounds]
        intensity of each spot in each channel, round.
    SpotCodeNo: float array [nSpots]
        Gene assigned to each spot.
    CharCodes: float array [nGenes x nRounds]
        Indicates which dye each gene contains in each round.
    BleedMatrix: float array [nChannels x nDyes x nRounds]
        Predicted intensity of each dye in each channel/round.
    MinSpots: integer
        Need more than this many spots to compute gene efficiency.
    GeneEfficiency: float [nGenes x nRounds].
        Expected intensity of each gene in each round as compared to the bleed matrix prediction.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_get_gene_efficiency(self):
        folder = os.path.join(self.folder, 'get_gene_efficiency')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            spot_colors, spot_gene_no, gene_codes, bleed_matrix, min_spots, output_matlab = \
                matlab.load_array(test_file, ['SpotColors', 'SpotCodeNo', 'CharCodes', 'BleedMatrix', 'MinSpots',
                                              'GeneEfficiency'])
            spot_colors = np.moveaxis(spot_colors, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            bleed_matrix = np.moveaxis(bleed_matrix, 2, 0).astype(float)  # change to r,c,d from MATLAB c,d,r
            spot_gene_no = spot_gene_no.flatten()-1  # python indices start at 0
            gene_codes = gene_codes.astype(int)
            min_spots = int(min_spots)
            output_python = get_gene_efficiency(spot_colors, spot_gene_no, gene_codes, bleed_matrix, min_spots)
            # ref_round has efficiency of 1.
            # Can only compare gene efficiencies with same ref_rounds and
            # if both matlab and python are above/below min_spots (number of spots is affected by choice of ref_round).
            ref_round_matlab = np.abs(output_matlab-1).argmin(axis=1)
            ref_round_python = np.abs(output_python-1).argmin(axis=1)
            use = np.logical_and(ref_round_matlab == ref_round_python,
                                 np.sum(output_matlab == 1,1) == np.sum(output_python == 1,1))
            if np.sum(use) == 0:
                raise ValueError('No data with same ref_round to compare.')
            diff = output_python[use] - output_matlab[use]
            self.assertTrue(np.abs(diff).max() <= self.tol)


class TestGetSpotIntensity(unittest.TestCase):
    """
    Check whether intensity calculated using jax and numpy is the same.
    """
    tol = 1e-6

    def test_get_spot_intensity(self):
        for i in range(5):
            n_spots = np.random.randint(3, 200)
            n_rounds = np.random.randint(1, 10)
            n_channels = np.random.randint(1, 10)
            spot_colors = (np.random.rand(n_spots, n_rounds, n_channels) - 0.5) * 2
            intensity = get_spot_intensity(spot_colors)
            intensity_jax = np.asarray(get_spot_intensity_vectorised(jnp.array(spot_colors)))
            diff = intensity - intensity_jax
            self.assertTrue(np.abs(diff).max() <= self.tol)
