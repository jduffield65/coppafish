import numpy as np

from coppafish.call_spots.dot_product_optimised import dot_product_score_no_weight, dot_product_score


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
