from coppafish.call_spots.background import fit_background
from coppafish.call_spots.base import get_non_duplicate, get_bled_codes, compute_gene_efficiency
from coppafish.call_spots.dot_product_optimised import dot_product_score_no_weight, dot_product_score
from coppafish.call_spots.qual_check_optimised import get_spot_intensity
import numpy as np
from jax import numpy as jnp


def test_get_non_duplicate():
    # Place two tiles
    tile_origin = np.array([[0,0,0],[100,0,0]])
    # Use both tiles
    use_tiles = [0,1]
    tile_centre = np.array([50,0,0], float)
    spot_local_yxz = np.array([[80,0,0], [110,0,0]], int)
    # Say each spot was found on the first tile
    spot_tile = np.array([0,0], int)
    output = get_non_duplicate(tile_origin, use_tiles, tile_centre, spot_local_yxz, spot_tile)
    assert spot_local_yxz.shape[0] == output.size, 'Expect output to have the same number of spots'
    assert output[0], 'First spot should be found closest to the first tile'
    assert not output[1], 'Second spot should be found closest to the other tile'


def test_get_bled_codes():
    # TODO: Flesh this unit test out to not just test output dimensions
    n_genes = 1
    n_rounds = 2
    n_channels = 3
    n_dyes = 4
    rng = np.random.RandomState(31)
    # Gene codes must be integers that exist inside bleed_matrix.shape[2]
    gene_codes = np.ones((n_genes, n_rounds), dtype=int)
    bleed_matrix = rng.random((n_rounds, n_channels, n_dyes))
    gene_efficiency = rng.random((n_genes, n_rounds))
    bled_codes = get_bled_codes(gene_codes, bleed_matrix, gene_efficiency)
    assert bled_codes.shape == (n_genes, n_rounds, n_channels), \
        'Expected (n_genes x n_rounds x n_channels) output shape'
    # TODO: Test this with a particular bleed matrix with a known output


def test_compute_gene_efficiency():
    # TODO: Flesh this unit test out to not just test output dimensions
    n_spots = 2
    n_rounds = 3
    n_channels = 4
    n_dyes = 5
    n_genes = 6
    rng = np.random.RandomState(71)
    spot_colours = rng.random((n_spots, n_rounds, n_channels))
    bled_codes = rng.randint(0, n_genes, size=(n_genes, n_rounds, n_channels))
    gene_no = rng.randint(0, n_genes, size=(n_spots))
    gene_score = rng.random((n_spots))
    gene_codes = rng.randint(0, n_dyes, size=(n_genes, n_rounds))
    intensity = rng.random(n_spots)
    gene_efficiency, use_ge, dye_efficiency = compute_gene_efficiency(spot_colours, bled_codes, gene_no, gene_score, gene_codes, intensity)
    # TODO: Finish this unit test


def test_fit_background():
    n_spots = 2
    n_rounds = 1
    n_channels = 2
    # rng = np.random.RandomState(0)
    spot_colours = np.empty((n_spots, n_rounds, n_channels))
    spot_colours[0,:,:] = 1
    spot_colours[1,:,:] = 0

    # Test with two spots, first one is just ones in every round and channel, the other is just zeroes everywhere.
    # The residual should then become just zeroes everywhere when the weight_shift is zero/super close to zero.
    residual1, coef1, background_vectors1 = fit_background(spot_colours, weight_shift=0)
    assert residual1.shape == spot_colours.shape, \
        'Expected outputted residual to be the same shape as the spot_colours array'
    assert coef1.shape == (n_spots, n_channels), 'Expected coefs to have shape n_spots x n_channels'
    assert background_vectors1.shape == (n_channels, n_rounds, n_channels), \
        'Expected coefs to have shape n_channels x n_spots x n_channels'
    assert np.allclose(residual1, 0), 'Expected all residuals to become zero after background fitting'
    # Create a random pattern of only background noise and check it removes it
    n_spots = 3
    n_rounds = 4
    n_channels = 5
    rng = np.random.RandomState(81)
    # Weighting given to background vector
    bg_weightings = rng.random((n_channels))
    spot_colours = np.ones((n_spots, n_rounds, n_channels))
    for c in range(n_channels):
        spot_colours[:,:,c] *= bg_weightings[c]
    residual1, coef1, background_vectors1 = fit_background(spot_colours, weight_shift=0)
    assert residual1.shape == spot_colours.shape, \
        'Expected outputted residual to be the same shape as the spot_colours array'
    assert coef1.shape == (n_spots, n_channels), 'Expected coefs to have shape n_spots x n_channels'
    assert background_vectors1.shape == (n_channels, n_rounds, n_channels), \
        'Expected coefs to have shape n_channels x n_spots x n_channels'
    assert np.allclose(residual1, 0), 'Expected all residuals to become near zero after background fitting'
    # Test weight_shift by seeing if it reduces the variance when increased
    spot_colours = rng.random((n_spots, n_rounds, n_channels)) * 10
    residual1, coef1, background_vectors1 = fit_background(spot_colours, weight_shift=0)
    residual2, coef2, background_vectors2 = fit_background(spot_colours, weight_shift=10)
    assert (residual1 != residual2).all(), 'Expecting different results when changing weight_shift'
    assert np.std(residual1) > np.std(residual2), 'Expected weight_shift to reduce the variance in the residuals'


def test_dot_product_score_no_weight():
    # Just use two spots, one round and one channel
    spot_colors = np.ones((2,1))
    # Set second spot colour to 0.5
    spot_colors[1,0] = 0.25
    # Use one gene
    n_genes = 1
    bled_codes = np.ones((n_genes,1))
    norm_shift = 0
    output = dot_product_score_no_weight(spot_colors, bled_codes, norm_shift)
    assert output.ndim == 2, 'Expected two dimensional array as output'
    assert output.shape[0] == spot_colors.shape[0], 'Expected second dimension to be spot count'
    assert output.shape[1] == n_genes, 'Expected second dimension to be the gene count'


def test_dot_product_score():
    # Dot product score but with the weighting matrix applied
    # Just use two spots, one round and one channel
    spot_colors = np.ones((2,1))
    # Set second spot colour to 0.5
    spot_colors[1,0] = 0.25
    # Use one gene
    n_genes = 1
    bled_codes = np.ones((n_genes,1))
    norm_shift = 0
    # Set the weighting to ones so change is applied
    weight_squared = np.ones((spot_colors.shape[0], spot_colors.shape[1]))
    output = dot_product_score(spot_colors, bled_codes, norm_shift, weight_squared)
    assert output.ndim == 2, 'Expected two dimensional array as output'
    assert output.shape[0] == spot_colors.shape[0], 'Expected second dimension to be spot count'
    assert output.shape[1] == n_genes, 'Expected second dimension to be the gene count'


def test_get_spot_intensity():
    n_spots = 2
    n_rounds = 3
    n_channels = 4
    spot_colors = np.zeros((n_spots, n_rounds, n_channels), dtype=float)
    spot_colors[0,0,0] = 1.
    spot_colors[1,0,3] = 2.
    spot_colors[1,1,3] = 2.
    output = get_spot_intensity(spot_colors)
    assert output.ndim == 1, 'Expect a vector output'
    assert output.size == n_spots, 'Expect vector dimensions to be n_spots'
    assert np.allclose(output[0], 0.), 'Expect first spot median to be zero'
    assert np.allclose(output[1], 2.), 'Expect second spot median to be 2'
