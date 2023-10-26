import pytest
import numpy as np


@pytest.mark.optimised
def test_dot_product_score_equality():
    from coppafish.call_spots.dot_product import dot_product_score
    from coppafish.call_spots.dot_product_optimised import dot_product_score as dot_product_score_jax

    rng = np.random.RandomState(7)
    n_spots = 2
    n_rounds = 7
    n_genes = 4
    n_channels_use = 3
    spot_colours = rng.rand(n_spots, n_rounds * n_channels_use)
    bled_codes = rng.rand(n_genes, n_rounds * n_channels_use)
    for weight_squared in [rng.rand(n_spots, n_rounds * n_channels_use), None]:
        for norm_shift in [rng.rand()]:
            gene_no, gene_score, gene_score_second, score = dot_product_score(spot_colours, bled_codes, weight_squared, 
                                                                       norm_shift)
            assert gene_no.shape == (n_spots, ), f'Expected `gene_no` to have shape ({n_spots}, )'
            assert gene_score.shape == (n_spots, ), f'Expected `gene_score` to have shape ({n_spots}, )'
            score_jax = dot_product_score_jax(spot_colours, bled_codes, norm_shift, weight_squared)
            assert score_jax.shape == (n_spots, n_genes), 'Unexpected dot product score shape'
            assert np.allclose(gene_score, np.max(score_jax, axis=1), atol=1e-4), 'Expected best scores to be similar'
            assert np.allclose(gene_no, np.argmax(score_jax, axis=1)), 'Expected the same best gene numbers'
            assert np.allclose(score, score_jax, atol=1e-4)
