import unittest
import os
import numpy as np
from ...utils import matlab, errors
from ..coefs_optimised import get_all_coefs, fit_coefs,\
    fit_coefs_weight, get_best_gene_first_iter, get_best_gene
from .. import coefs as no_jax
import jax.numpy as jnp


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
    tol = 1e-6

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
            coefs_matlab = coefs_matlab.flatten()  # change to g,s from MATLAB s,g.
            n_genes = bled_codes.shape[0]
            n_spots, n_rounds, n_channels = spot_colors.shape

            bc_use = bled_codes.reshape(n_genes, -1).transpose()
            sc_use = spot_colors.reshape(n_spots, -1).transpose()

            genes_used = np.tile(np.expand_dims(np.arange(n_genes), 0), (n_spots, 1))
            residual_python, coefs_python = no_jax.fit_coefs(bc_use, sc_use, genes_used)
            residual_jax, coefs_jax = fit_coefs(bc_use, sc_use, genes_used)
            diff1_jax = np.asarray(residual_jax) - residual_python
            diff2_jax = np.asarray(coefs_jax) - coefs_python
            residual_jax = np.asarray(residual_jax).reshape(n_spots, n_rounds, n_channels)
            coefs_jax = np.asarray(coefs_jax).flatten()
            diff1 = residual_jax - residual_matlab
            diff2 = coefs_jax - coefs_matlab
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)
            self.assertTrue(np.abs(diff1_jax).max() <= self.tol)
            self.assertTrue(np.abs(diff2_jax).max() <= self.tol)

            # Check weighted least squares works too.
            weight = np.random.rand(n_spots, n_channels*n_rounds)
            residual_python_weight, coefs_python_weight = \
                no_jax.fit_coefs_weight(bled_codes.reshape(n_genes, -1).transpose(),
                                        spot_colors.reshape(n_spots, -1).transpose(), genes_used, weight)
            residual_jax_weight, coefs_jax_weight = \
                fit_coefs_weight(bled_codes.reshape(n_genes, -1).transpose(),
                                 spot_colors.reshape(n_spots, -1).transpose(), genes_used,
                                 weight)
            residual_jax_weight = np.asarray(residual_jax_weight)
            coefs_jax_weight = np.asarray(coefs_jax_weight)
            diff1_weight = residual_python_weight - residual_jax_weight
            diff2_weight = coefs_python_weight - coefs_jax_weight
            self.assertTrue(np.abs(diff1_weight).max() <= self.tol)
            self.assertTrue(np.abs(diff2_weight).max() <= self.tol)


class TestGetBestGene(unittest.TestCase):
    """
    Check whether jax and non-jax get_best_gene functions give the same results
    """
    N_Tests = 5
    tol = 1e-4
    MinPixels = 5
    MaxPixels = 100
    MinChannels = 2
    MaxChannels = 9
    MinRounds = 2
    MaxRounds = 9
    MinGenes = MaxChannels * 2
    MaxGenes = MaxChannels * 5
    MinGenesAdded = 1
    MaxGenesAdded = MaxChannels
    Beta = 1
    MinAlpha = 70
    MaxAlpha = 200
    MinNormShift = 0.001
    MaxNormShift = 0.1
    MinScoreThresh = 0.1
    MaxScoreThresh = 0.3

    def get_params(self):
        n_pixels = np.random.randint(self.MinPixels, self.MaxPixels)
        n_channels = np.random.randint(self.MinChannels, self.MaxChannels)
        n_rounds = np.random.randint(self.MinRounds, self.MaxRounds)
        n_genes = np.random.randint(self.MinGenes, self.MaxGenes)
        pixel_colors = np.random.uniform(-1, 1, (n_pixels, n_rounds * n_channels))
        pixel_colors = pixel_colors / np.linalg.norm(pixel_colors, axis=1, keepdims=True)  # L2 norm of 1
        bled_codes = np.random.uniform(-1, 1, (n_genes, n_rounds * n_channels))
        bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=1, keepdims=True)  # L2 norm of 1
        background_coefs = np.random.uniform(-1, 1, (n_pixels, n_channels))
        norm_shift = np.random.uniform(self.MinNormShift, self.MaxNormShift)
        score_thresh = np.random.uniform(self.MinScoreThresh, self.MaxScoreThresh)
        alpha = np.random.uniform(self.MinAlpha, self.MaxAlpha)
        beta = self.Beta
        background_genes = np.arange(n_genes-n_channels, n_genes)

        # Stuff for not 1st iter
        n_genes_added = np.random.randint(self.MinGenesAdded, self.MaxGenesAdded)
        genes_added = np.zeros((n_pixels, n_genes_added), dtype=int)
        for i in range(n_pixels):
            # Must have no duplicate genes for each pixel and must not include background genes
            genes_added[i] = np.random.choice(n_genes - n_channels, n_genes_added, replace=False)
        gene_coefs = np.random.uniform(-1, 1, (n_pixels, n_genes_added))
        return pixel_colors, bled_codes, background_coefs, norm_shift, score_thresh, alpha, beta, background_genes, \
               genes_added, gene_coefs

    def test_first_iter(self):
        for i in range(self.N_Tests):
            pixel_colors, bled_codes, background_coefs, norm_shift, score_thresh, alpha, beta, \
                background_genes, _, _ = self.get_params()
            best_gene_jax, pass_score_thresh_jax, background_var_jax = \
                get_best_gene_first_iter(pixel_colors, bled_codes, background_coefs, norm_shift,
                                         score_thresh, alpha, beta, background_genes)
            best_gene, pass_score_thresh, background_var, best_score = \
                no_jax.get_best_gene_first_iter(pixel_colors, bled_codes, background_coefs, norm_shift,
                                                score_thresh, alpha, beta, background_genes)
            diff1 = best_gene - np.asarray(best_gene_jax)
            diff2 = pass_score_thresh.astype(int) - np.asarray(pass_score_thresh_jax).astype(int)
            diff3 = background_var - np.asarray(background_var_jax)
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)
            self.assertTrue(np.abs(diff3).max() <= self.tol)

    def test_get_best_gene(self):
        for i in range(self.N_Tests):
            pixel_colors, bled_codes, background_coefs, norm_shift, score_thresh, alpha, beta, \
                background_genes, genes_added, gene_coefs = self.get_params()
            background_var = \
                no_jax.get_best_gene_first_iter(pixel_colors, bled_codes, background_coefs, norm_shift,
                                                score_thresh, alpha, beta, background_genes)[2]
            best_gene_jax, pass_score_thresh_jax, inverse_var_jax = \
                get_best_gene(pixel_colors, bled_codes, gene_coefs, genes_added, norm_shift,
                              score_thresh, alpha, background_genes, background_var)
            best_gene, pass_score_thresh, inverse_var, best_score = \
                no_jax.get_best_gene(pixel_colors, bled_codes, gene_coefs, genes_added, norm_shift,
                                     score_thresh, alpha, background_genes, background_var)
            diff1 = best_gene - np.asarray(best_gene_jax)
            diff2 = pass_score_thresh.astype(int) - np.asarray(pass_score_thresh_jax).astype(int)
            diff3 = inverse_var - np.asarray(inverse_var_jax)
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)
            self.assertTrue(np.abs(diff3).max() <= self.tol)


class TestGetAllCoefs(unittest.TestCase):
    """
    Check whether get_all_coefs works the same as MATLAB function:
    iss-Josh/@iss_OMP/get_omp_coefs.
    Data saved with script:
    python_testing/omp/get_omp_coefs.m

    test files contain:
    bled_codes: float array [nGenes x nChannels x nRounds]
        Expected code for each gene including gene_efficiency information.
    spot_colors: float array [nSpots x nChannels x nRounds]
        Code of each spot.
    background_shift: float
        shift to apply to background weighting to stop blow up to infinity.
    dp_shift: float
        shift to apply to normalisation of SpotColors when taking dot product i.e. weak Spots get lower score.
        In MATLAB, this is o.ompWeightShift * sqrt(nRounds) where o.ompWeightShift is the shift per round.
    alpha: float
        constant, by how much to increase variance as genes added.
    beta: float
        standard deviation in image with no genes added.
    max_genes: integer
        maximum number of genes added to each pixel.
    dp_thresh: float
        DotProductScore must exceed this for best gene for that gene to be added.
        In MATLAB, each round has a max score of 1 so max dot product is nRounds.
    mod_fit: bool
        True if fit coefficients via weighted least squares using 1/sigma as the weighting.
    FinalCoefs: float array [nSpots x nGenes]
        Coefficient found for each gene for each pixel.
    BackgroundCoef: float array [nSpots x nChannels].
        Coefficient found for each background vector for each pixel.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol_matlab = 1e-3
    tol_python = 1e-5

    def test_get_all_coefs(self):
        folder = os.path.join(self.folder, 'get_all_coefs')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            spot_colors, bled_codes, background_shift, dp_shift, alpha, beta, max_genes, dp_thresh, weight_coef_fit,\
            coefs_matlab, background_coefs_matlab = \
                matlab.load_array(test_file, ['spot_colors', 'bled_codes', 'background_shift', 'dp_shift', 'alpha',
                                              'beta', 'max_genes', 'dp_thresh', 'mod_fit',
                                              'FinalCoefs','BackgroundCoef'])
            bled_codes = jnp.array(np.moveaxis(bled_codes, 1, 2).astype(float))  # change to r,c from MATLAB c,r
            spot_colors = jnp.array(np.moveaxis(spot_colors, 1, 2).astype(float))  # change to r,c from MATLAB c,r
            bled_codes = bled_codes / np.expand_dims(np.linalg.norm(bled_codes, axis=(1, 2)), (1, 2)) # make norm of all codes 1.
            n_rounds = spot_colors.shape[1]

            # matlab background vectors do not have L2 norm = 1 but all non-zero elements are 1 so easy to compensate
            matlab_norm_factor = np.linalg.norm(np.ones(n_rounds))
            background_coefs_matlab = background_coefs_matlab * matlab_norm_factor

            max_genes = int(max_genes)
            background_shift = float(background_shift)
            dp_shift = float(dp_shift)
            dp_thresh = float(dp_thresh) / n_rounds  # in python, max dot product is 1 not nRounds.
            alpha = float(alpha)
            beta = float(beta)
            weight_coef_fit = bool(weight_coef_fit)
            coefs_jax, background_coefs_jax = get_all_coefs(spot_colors, bled_codes, background_shift, dp_shift,
                                                            dp_thresh, alpha, beta, max_genes, weight_coef_fit)
            coefs_python, background_coefs_python = \
                no_jax.get_all_coefs(np.asarray(spot_colors), np.asarray(bled_codes), background_shift, dp_shift,
                                     dp_thresh, alpha, beta, max_genes, weight_coef_fit)
            diff1_python = coefs_jax - coefs_python
            diff2_python = background_coefs_jax - background_coefs_python
            diff1 = coefs_jax - coefs_matlab
            diff2 = background_coefs_jax - background_coefs_matlab
            # make sure same coefficients non-zero for both methods.
            diff_nnz_coefs = np.abs(np.sum((coefs_jax != 0) != (coefs_matlab !=0), axis=1))
            self.assertTrue(np.abs(diff1_python).max() <= self.tol_python)
            self.assertTrue(np.abs(diff2_python).max() <= self.tol_python)
            self.assertTrue((diff_nnz_coefs > 0).sum() / diff_nnz_coefs.size < 0.01)
            self.assertTrue(np.abs(diff1[diff_nnz_coefs == 0]).max() <= self.tol_matlab)
            self.assertTrue(np.abs(diff2).max() <= self.tol_matlab)
