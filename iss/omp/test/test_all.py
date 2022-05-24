import unittest
import os
import numpy as np
from ...utils import matlab, errors
from ..base import fitting_standard_deviation, fit_coefs, get_all_coefs, fit_coefs_jax,\
    fit_coefs_weight_jax
from ..cython_omp import mat_mul, cy_fit_coefs
from ..spots import count_spot_neighbours
import jax


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
    tol = 1e-6

    def test_mat_mul(self):
        m = np.random.randint(100)
        n = np.random.randint(100)
        a = np.random.rand(m, n)
        x = np.random.rand(n)
        out_numpy = a @ x
        out_cython = np.zeros(m)
        # For some reason, need to transpose a twice here but not in fit_coefs.
        mat_mul(a.transpose(), x, out_cython, transpose='t'.encode('utf-8'))
        diff = out_numpy - out_cython
        self.assertTrue(np.abs(diff).max() <= self.tol)

    def test_fit_coefs(self):
        folder = os.path.join(self.folder, 'fit_coefs')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        batch_fit_coefs_jax = jax.vmap(fit_coefs_jax, in_axes=(None, 1, 0), out_axes=(1, 0))
        batch_fit_coefs_weight_jax = jax.vmap(fit_coefs_weight_jax, in_axes=(None, 1, 0, 1), out_axes=(1, 0))
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
            if n_spots == 1:
                sc_use_python = sc_use.flatten()
            else:
                sc_use_python = sc_use
            genes_used = np.tile(np.expand_dims(np.arange(n_genes), 0), (n_spots, 1))
            # if n_spots > 1:
            #     residual_python, coefs_python = fit_coefs_single_gene(bc_use, sc_use_python, genes_used.flatten())
            # else:
            #     residual_python, coefs_python = fit_coefs_multi_genes(bc_use, sc_use, genes_used)
            residual_python, coefs_python = fit_coefs(bc_use, sc_use_python)
            residual_python = residual_python.transpose().reshape(n_spots, n_rounds, n_channels)
            #genes_used = np.tile(np.expand_dims(np.arange(n_genes), 0), (n_spots, 1))
            residual_jax, coefs_jax = batch_fit_coefs_jax(bc_use, sc_use, genes_used)
            residual_jax = np.asarray(residual_jax).transpose().reshape(n_spots, n_rounds, n_channels)
            coefs_jax = np.asarray(coefs_jax).flatten()
            residual_cython, coefs_cython = cy_fit_coefs(bc_use, sc_use, genes_used)
            coefs_cython = coefs_cython.flatten()
            residual_cython = residual_cython.transpose().reshape(n_spots, n_rounds, n_channels)
            diff1 = residual_python - residual_matlab
            diff2 = coefs_python - coefs_matlab
            diff1_jax = residual_python - residual_jax
            diff2_jax = coefs_python - coefs_jax
            diff1_cython = residual_cython - residual_python
            diff2_cython = coefs_python - coefs_cython
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)
            self.assertTrue(np.abs(diff1_jax).max() <= self.tol)
            self.assertTrue(np.abs(diff2_jax).max() <= self.tol)
            self.assertTrue(np.abs(diff1_cython).max() <= self.tol)
            self.assertTrue(np.abs(diff2_cython).max() <= self.tol)
        if n_spots == 1:
            # Check weighted least squares works if only 1 spot. If more than 1 spot, fit_coefs won't work.
            weight = np.random.rand(n_channels*n_rounds, 1)
            residual_python_weight, coefs_python_weight = fit_coefs(bc_use, sc_use_python, weight.flatten())
            residual_python_weight = residual_python_weight.transpose().reshape(n_spots, n_rounds, n_channels)
            residual_jax_weight, coefs_jax_weight = \
                batch_fit_coefs_weight_jax(bled_codes.reshape(n_genes, -1).transpose(),
                                           spot_colors.reshape(n_spots, -1).transpose(), genes_used, weight)
            residual_jax_weight = np.asarray(residual_jax_weight).transpose().reshape(n_spots, n_rounds, n_channels)
            coefs_jax_weight = np.asarray(coefs_jax_weight)
            diff1_weight = residual_python_weight - residual_jax_weight
            diff2_weight = coefs_python_weight - coefs_jax_weight
            self.assertTrue(np.abs(diff1_weight).max() <= self.tol)
            self.assertTrue(np.abs(diff2_weight).max() <= self.tol)


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
    tol = 1e-6

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
            bled_codes = np.moveaxis(bled_codes, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            spot_colors = np.moveaxis(spot_colors, 1, 2).astype(float)  # change to r,c from MATLAB c,r
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
            coefs_python, background_coefs_python = get_all_coefs(spot_colors, bled_codes, background_shift, dp_shift,
                                                                  dp_thresh, alpha, beta, max_genes, weight_coef_fit)
            diff1 = coefs_python - coefs_matlab
            diff2 = background_coefs_python - background_coefs_matlab
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)


class TestCountSpotNeighbours(unittest.TestCase):
    """
    Check whether count_spot_neighbours works the same as MATLAB function:
    iss-Josh/@iss_OMP/detect_peak_genes_omp.
    Data saved with script:
    python_testing/omp/get_spot_neighbours.m

    test files contain:
    GeneIm: float array [nY x nX]
        Gene coefficient image.
    PeakYXZ: int array [nSpots x 3]
        YXZ coordinate of spots found. Z is always 1.
    PosFilter: int array [nFilterY x nFilterX]
        Filter indicates region about each spot we expect positive coefficients.
    NegFilter: int array [nFilterY x nFilterX]
        Filter indicates region about each spot we expect negative coefficients.
    nPosNeighb: int array [nSpots]
        Number of positive neighbours about each spot.
    nNegNeighb: int array [nSpots]
        Number of negative neighbours about each spot.
    """
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    tol = 1e-10

    def test_count_spot_neighbours(self):
        folder = os.path.join(self.folder, 'count_spot_neighbours')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            image, spot_yxz, pos_filter, neg_filter, pos_neighb_matlab, neg_neighb_matlab = \
                matlab.load_array(test_file, ['GeneIm', 'PeakYXZ', 'PosFilter', 'NegFilter',
                                              'nPosNeighb', 'nNegNeighb'])
            spot_yxz = (spot_yxz - 1).astype(int)  # MATLAB to python indexing.
            # In MATLAB, did 'symmetric' padding but in Python, doing 0 padding
            # (i.e. assuming all coefficients are 0 outside image).
            pad_size = [(int((ax_size - 1) / 2),) * 2 for ax_size in pos_filter.shape]
            spot_yxz_pad = spot_yxz.copy()
            for i in range(len(pad_size)):
                spot_yxz_pad[:, i] = spot_yxz[:,i] + pad_size[i][0]
            if (pos_filter + neg_filter).max() > 1:
                raise ValueError('Pos and Neg overlap')
            pos_neighb_python, neg_neighb_python = count_spot_neighbours(np.pad(image, pad_size, 'symmetric'),
                                                                         spot_yxz_pad, pos_filter-neg_filter)
            diff1 = pos_neighb_python - pos_neighb_matlab.squeeze()
            diff2 = neg_neighb_python - neg_neighb_matlab.squeeze()
            self.assertTrue(np.abs(diff1).max() <= self.tol)
            self.assertTrue(np.abs(diff2).max() <= self.tol)
