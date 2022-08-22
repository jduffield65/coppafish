import os
import unittest
import numpy as np
from ...omp import count_spot_neighbours
from ...omp.coefs import get_best_gene_first_iter, get_best_gene
from ...utils import errors, matlab


def fitting_variance(bled_codes: np.ndarray, coef: np.ndarray, alpha: float, beta: float = 1) -> np.ndarray:
    """
    Old method before Jax.
    Based on maximum likelihood estimation, this finds the variance accounting for all genes fit in
    each round/channel. The more genes added, the greater the variance so if the inverse is used as a
    weighting for omp fitting, the rounds/channels which already have genes in will contribute less.

    Args:
        bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        coef: `float [n_pixels x n_genes]`.
            Coefficient of each `bled_code` for each pixel found on the previous OMP iteration.
        alpha: By how much to increase variance as genes added.
        beta: The variance with no genes added (`coef=0`) is `beta**2`.

    Returns:
        `float [n_pixels x (n_rounds x n_channels)]`
            Standard deviation of each pixel in each round/channel based on genes fit.
    """
    n_genes = bled_codes.shape[0]
    n_pixels = coef.shape[0]
    if not errors.check_shape(coef, [n_pixels, n_genes]):
        raise errors.ShapeError('coef', coef.shape, (n_pixels, n_genes))

    var = (coef**2 @ bled_codes**2) * alpha + beta ** 2

    # # Old method - much slower
    # n_genes, n_rounds, n_channels = bled_codes.shape
    # var = np.ones((n_pixels, n_rounds, n_channels)) * beta ** 2
    # for g in range(n_genes):
    #     var = var + alpha * np.expand_dims(coef[:, g] ** 2, (1, 2)) * np.expand_dims(bled_codes[g] ** 2, 0)

    return var


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
    tol = 1e-5

    def test_fitting_standard_deviation(self):
        folder = os.path.join(self.folder, 'fitting_standard_deviation')
        test_files = [s for s in os.listdir(folder) if "test" in s]
        if len(test_files) == 0:
            raise errors.EmptyListError("test_files")
        for file_name in test_files:
            test_file = os.path.join(folder, file_name)
            bled_codes, coef, alpha, beta, output_matlab = \
                matlab.load_array(test_file, ['bled_codes', 'coef', 'alpha', 'beta', 'sigma'])
            n_pixels, n_genes = coef.shape
            bled_codes = np.moveaxis(bled_codes, 1, 2).astype(float).reshape(n_genes, -1)  # change to r,c from MATLAB c,r
            n_round_channels = bled_codes.shape[1]
            # spot_colors is placeholder, needed for function but not variance calc.
            spot_colors = np.zeros((n_pixels, n_round_channels))
            output_matlab = np.moveaxis(output_matlab, 1, 2).astype(float)  # change to r,c from MATLAB c,r
            genes_added = np.where(coef != 0)[1].reshape(n_pixels, -1)
            # genes present for all pixels are background genes.
            columns_all_genes_same = np.all(genes_added == genes_added[0, :], axis=0)
            background_genes = genes_added[0, columns_all_genes_same]
            # Compute background contribution to variance first as background_coefs do not change.
            background_var = get_best_gene_first_iter(
                spot_colors, bled_codes, coef[:, background_genes], 0, 0,
                float(alpha), float(beta), background_genes)[2]

            if columns_all_genes_same.all():
                # all background, computation ends here.
                output_python = np.sqrt(background_var)
            else:
                # If genes other than background, then add this contribution.
                actual_genes = genes_added[:, np.invert(columns_all_genes_same)]
                gene_coefs = coef[np.arange(n_pixels)[:, np.newaxis], actual_genes]
                inverse_var = get_best_gene(
                    spot_colors, bled_codes, gene_coefs, actual_genes, 0, 0,
                    float(alpha), background_genes, background_var)[2]
                output_python = np.sqrt(1/inverse_var)

            output_python_no_jax = np.sqrt(fitting_variance(bled_codes, coef, float(alpha), float(beta)))
            diff = output_python - output_matlab.reshape(n_pixels, -1)
            diff2 = output_python_no_jax - output_python
            self.assertTrue(np.abs(diff).max() <= self.tol)
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
            diff_pos1 = pos_neighb_python - pos_neighb_matlab.squeeze()
            diff_neg1 = neg_neighb_python - neg_neighb_matlab.squeeze()
            # Check if matched MATLAB
            self.assertTrue(np.abs(diff_pos1).max() <= self.tol)
            self.assertTrue(np.abs(diff_neg1).max() <= self.tol)
