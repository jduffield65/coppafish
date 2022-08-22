import unittest
import os
import numpy as np
from ...utils import matlab, errors
from ..dot_product import dot_product_score, dot_product_score_no_weight
from ..dot_product_optimised import dot_product_score as dot_product_score_optimised
from ..dot_product_optimised import dot_product_score_no_weight as dot_product_score_no_weight_optimised
from ..background import fit_background
from ..background_optimised import fit_background as fit_background_optimised
from ..qual_check import get_spot_intensity
from ..qual_check_optimised import get_spot_intensity as get_spot_intensity_optimised
import jax.numpy as jnp


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
    tol = 1e-6

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
            n_spots, n_rounds, n_channels = spot_colors.shape
            spot_colors = spot_colors.reshape(-1, n_rounds * n_channels)
            bled_codes = bled_codes.reshape(-1, n_rounds * n_channels)
            if weight.max() == 0:
                # empty array in Matlab loaded as [0, 0] - this refers to weight_squared = None case.
                output_python = dot_product_score_no_weight(spot_colors, bled_codes, norm_shift)
                output_jax = np.asarray(dot_product_score_no_weight_optimised(spot_colors, bled_codes, norm_shift))
            else:
                weight_squared = np.moveaxis(weight, 1, 2).astype(float) ** 2 # change to r,c from MATLAB c,r
                weight_squared = weight_squared.reshape(-1, n_rounds * n_channels)
                output_python = dot_product_score(spot_colors, bled_codes, norm_shift, weight_squared)
                output_jax = np.asarray(dot_product_score_optimised(spot_colors, bled_codes, norm_shift,
                                                                    weight_squared))
            diff = output_python - output_matlab
            diff_jax = output_python - output_jax
            self.assertTrue(np.abs(diff).max() <= self.tol)
            self.assertTrue(np.abs(diff_jax).max() <= self.tol)


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
            residual_jax, coef_jax, background_vectors_jax = fit_background_optimised(spot_colors, weight_shift)
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
            intensity_jax = np.asarray(get_spot_intensity_optimised(jnp.array(spot_colors)))
            diff = intensity - intensity_jax
            self.assertTrue(np.abs(diff).max() <= self.tol)
