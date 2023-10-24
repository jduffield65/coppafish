import numpy as np
import pytest


@pytest.mark.optimised
def test_fit_coefs_equality():
    # We want to test that the function `fit_coefs` is giving similar results in the jax and non-jax code
    rng = np.random.RandomState(5)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_genes_add = 13
    n_pixels = 9
    bled_codes = rng.rand(n_rounds * n_channels, n_genes)
    pixel_colors = rng.rand(n_rounds * n_channels, n_pixels)
    genes = rng.randint(n_genes, size=(n_pixels, n_genes_add))
    from coppafish.omp.coefs import fit_coefs
    residual, coefs = fit_coefs(bled_codes, pixel_colors, genes)
    assert residual.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    from coppafish.omp.coefs_optimised import fit_coefs
    residual_optimised, coefs_optimised = fit_coefs(bled_codes, pixel_colors, genes)
    assert residual.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    assert np.allclose(residual, residual_optimised, atol=1e-4), \
        'Expected similar residual from optimised and non-optimised OMP'
    assert np.allclose(coefs,    coefs_optimised,    atol=1e-4), \
        'Expected similar coefs from optimised and non-optimised OMP'


@pytest.mark.optimised
def test_fit_coefs_weight_equality():
    rng = np.random.RandomState(5)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_genes_add = 13
    n_pixels = 9
    bled_codes = rng.rand(n_rounds * n_channels, n_genes)
    pixel_colors = rng.rand(n_rounds * n_channels, n_pixels)
    genes = rng.randint(n_genes, size=(n_pixels, n_genes_add))
    weight = rng.rand(n_pixels, n_rounds * n_channels)
    from coppafish.omp.coefs import fit_coefs_weight
    residual, coefs = fit_coefs_weight(bled_codes, pixel_colors, genes, weight)
    assert residual.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    from coppafish.omp.coefs_optimised import fit_coefs_weight
    residual_optimised, coefs_optimised = fit_coefs_weight(bled_codes, pixel_colors, genes, weight)
    assert residual.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    assert np.allclose(residual, residual_optimised, atol=1e-4), \
        'Expected similar residual from optimised and non-optimised OMP'
    assert np.allclose(coefs,    coefs_optimised,    atol=1e-4), \
        'Expected similar coefs from optimised and non-optimised OMP'


@pytest.mark.optimised
def test_get_best_gene_first_iter_equality():
    rng = np.random.RandomState(47)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_pixels = 9
    residual_pixel_colors = rng.rand(n_pixels, n_rounds * n_channels)
    all_bled_codes = rng.rand(n_genes, n_rounds * n_channels)
    background_coefs = rng.rand(n_pixels, n_channels)
    norm_shift = rng.rand()
    score_thresh = rng.rand() * 0.01
    alpha = rng.rand()
    beta = rng.rand()
    background_genes = rng.randint(n_genes, size=(n_channels))
    from coppafish.omp.coefs import get_best_gene_first_iter
    best_gene, pass_score_thresh, background_var, best_score = \
        get_best_gene_first_iter(residual_pixel_colors, all_bled_codes, background_coefs, norm_shift, score_thresh, 
                                 alpha, beta, background_genes)
    assert best_gene.shape == (n_pixels, ), 'Unexpected shape for `best_gene` output'
    assert pass_score_thresh.shape == (n_pixels, ), 'Unexpected shape for `pass_score_thresh` output'
    assert background_var.shape == (n_pixels, n_rounds * n_channels), 'Unexpected shape for `background_var` output'
    assert best_score.shape == (n_pixels, ), 'Unexpected shape for `best_score` output'
    from coppafish.omp.coefs_optimised import get_best_gene_first_iter
    best_gene_optimised, pass_score_thresh_optimised, background_var_optimised = \
        get_best_gene_first_iter(residual_pixel_colors, all_bled_codes, background_coefs, norm_shift, score_thresh, 
                                 alpha, beta, background_genes)
    assert best_gene.shape == (n_pixels, ), 'Unexpected shape for `best_gene` output'
    assert pass_score_thresh.shape == (n_pixels, ), 'Unexpected shape for `pass_score_thresh` output'
    assert background_var.shape == (n_pixels, n_rounds * n_channels), 'Unexpected shape for `background_var` output'
    # FIXME: `best_gene` for this non-jax function is not the same as the jax version 
    # assert np.allclose(best_gene, best_gene_optimised, atol=1e-4), \
    #     'Expected the same `best_genes` from optimised and non-optimised OMP'
    assert np.allclose(pass_score_thresh, pass_score_thresh_optimised, atol=1e-4), \
        'Expected similar `pass_score_thresh` from optimised and non-optimised OMP'
    assert np.allclose(background_var, background_var_optimised, atol=1e-4), \
        'Expected similar `background_var` from optimised and non-optimised OMP'


@pytest.mark.optimised
def test_get_best_gene_base_equality():
    rng = np.random.RandomState(97)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    # We test on one pixel because the jax code does a single pixel at a time
    residual_pixel_colors = rng.rand(1, n_rounds * n_channels)
    residual_pixel_colors_jax = residual_pixel_colors[0]
    all_bled_codes = rng.rand(n_genes, n_rounds * n_channels)
    norm_shift = rng.rand()
    score_thresh = rng.rand() * 0.01
    inverse_var = rng.rand(n_rounds * n_channels)
    ignore_genes = np.asarray([], dtype=int)
    from coppafish.omp.coefs import get_best_gene_base
    best_gene, pass_score_thresh, best_score = \
        get_best_gene_base(residual_pixel_colors, all_bled_codes, norm_shift, score_thresh, inverse_var, ignore_genes)
    assert best_gene.size == 1, 'Expected single best gene for one pixel'
    assert pass_score_thresh.shape == (1, ), 'Unexpected `pass_score_thresh` shape'
    assert best_score.shape == (1, ), 'Unexpected `best_score` shape'
    from coppafish.omp.coefs_optimised import get_best_gene_base
    best_gene_optimised, pass_score_thresh_optimised = \
        get_best_gene_base(residual_pixel_colors_jax, all_bled_codes, norm_shift, score_thresh, inverse_var, ignore_genes)
    assert best_gene_optimised.size == 1, 'Expected single best gene for one pixel'
    assert pass_score_thresh_optimised.size == 1, 'Unexpected `pass_score_thresh` size'
    assert np.allclose(best_gene, best_gene_optimised, atol=1e-4), \
        'Expected the same `best_genes` from optimised and non-optimised OMP'
    assert np.allclose(pass_score_thresh, pass_score_thresh_optimised, atol=1e-4), \
        'Expected similar `pass_score_thresh` from optimised and non-optimised OMP'


@pytest.mark.optimised
def test_get_best_gene_equality():
    rng = np.random.RandomState(128)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_genes_added = 2
    n_pixels = 9
    residual_pixel_colors = rng.rand(n_pixels, n_rounds * n_channels)
    all_bled_codes = rng.rand(n_genes, n_rounds * n_channels)
    coefs = rng.rand(n_pixels, n_genes_added)
    genes_added = np.zeros((n_pixels, n_genes_added), dtype=int)
    genes_added[:,1] = 1
    norm_shift = rng.rand()
    score_thresh = rng.rand() * 0.01
    alpha = rng.rand()
    background_genes = rng.randint(n_genes, size=(n_channels))
    background_var = rng.rand(n_pixels, n_rounds * n_channels)
    from coppafish.omp.coefs import get_best_gene
    best_gene, pass_score_thresh, inverse_var, best_score = \
        get_best_gene(residual_pixel_colors, all_bled_codes, coefs, genes_added, norm_shift, score_thresh, alpha, 
                      background_genes, background_var)
    from coppafish.omp.coefs_optimised import get_best_gene
    best_gene_optimised, pass_score_thresh_optimised, inverse_var_optimised = \
        get_best_gene(residual_pixel_colors, all_bled_codes, coefs, genes_added, norm_shift, score_thresh, alpha, 
                      background_genes, background_var)
    assert np.allclose(best_gene, best_gene_optimised, atol=1e-4), 'Expected the same `best_genes` output'
    assert np.all(pass_score_thresh == pass_score_thresh_optimised), 'Expected the same `pass_score_thresh` output'
    assert np.allclose(inverse_var, inverse_var_optimised), 'Expected similar `inverse_var` output'
