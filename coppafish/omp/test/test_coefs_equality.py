import numpy as np
import pytest


@pytest.mark.optimised
def test_fit_coefs_equality():
    # We want 1 test that the function `fit_coefs` is giving similar results in the jax and non-jax code
    rng = np.random.RandomState(9)
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
    rng = np.random.RandomState(34)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    n_genes_add = 5
    n_pixels = 9
    bled_codes = rng.rand(n_rounds * n_channels, n_genes) + 1
    pixel_colors = rng.rand(n_rounds * n_channels, n_pixels) + 1
    genes = np.arange(n_genes_add, dtype=int)
    genes = np.repeat([genes], n_pixels, axis=0)
    weight = rng.rand(n_pixels, n_rounds * n_channels) + 10
    bled_codes.astype(np.float32)
    pixel_colors.astype(np.float32)
    weight.astype(np.float32)
    from coppafish.omp.coefs import fit_coefs_weight
    residual, coefs = fit_coefs_weight(bled_codes, pixel_colors, genes, weight)
    assert residual.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    from coppafish.omp.coefs_optimised import fit_coefs_weight
    residual_optimised, coefs_optimised = fit_coefs_weight(bled_codes, pixel_colors, genes, weight)
    residual_optimised = np.asarray(residual_optimised, dtype=np.float32)
    coefs_optimised = np.asarray(coefs_optimised, dtype=np.float32)
    assert residual_optimised.shape == (n_pixels, n_rounds * n_channels), 'Unexpected output residual shape'
    assert coefs_optimised.shape == (n_pixels, n_genes_add), 'Unexpected output coefs shape'
    #FIXME: This sometimes caused an assertion error in a random version of python sometimes.... no idea what is 
    # happening. Maybe random float rounding is causing this??
    assert np.allclose(residual, residual_optimised, atol=1e-3), \
        'Expected similar residual from optimised and non-optimised OMP'
    assert np.allclose(coefs,    coefs_optimised,    atol=1e-3), \
        'Expected similar coefs from optimised and non-optimised OMP'


@pytest.mark.optimised
def test_get_best_gene_first_iter_equality():
    from coppafish.omp.coefs import get_best_gene_first_iter
    from coppafish.omp.coefs_optimised import get_best_gene_first_iter as get_best_gene_first_iter_jax

    rng = np.random.RandomState(60)
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
    best_gene, pass_score_thresh, background_var = \
        get_best_gene_first_iter(residual_pixel_colors, all_bled_codes, background_coefs, norm_shift, score_thresh, 
                                 alpha, beta, background_genes)
    assert best_gene.shape == (n_pixels, ), 'Unexpected shape for `best_gene` output'
    assert pass_score_thresh.shape == (n_pixels, ), 'Unexpected shape for `pass_score_thresh` output'
    assert background_var.shape == (n_pixels, n_rounds * n_channels), 'Unexpected shape for `background_var` output'
    best_gene_optimised, pass_score_thresh_optimised, background_var_optimised = \
        get_best_gene_first_iter_jax(residual_pixel_colors, all_bled_codes, background_coefs, norm_shift, score_thresh, 
                                     alpha, beta, background_genes)
    assert best_gene.shape == (n_pixels, ), 'Unexpected shape for `best_gene` output'
    assert pass_score_thresh.shape == (n_pixels, ), 'Unexpected shape for `pass_score_thresh` output'
    assert background_var.shape == (n_pixels, n_rounds * n_channels), 'Unexpected shape for `background_var` output'
    assert np.allclose(best_gene, best_gene_optimised, atol=1e-4), \
        'Expected the same `best_genes` from optimised and non-optimised OMP'
    assert np.allclose(pass_score_thresh, pass_score_thresh_optimised, atol=1e-4), \
        'Expected similar `pass_score_thresh` from optimised and non-optimised OMP'
    assert np.allclose(background_var, background_var_optimised, atol=1e-4), \
        'Expected similar `background_var` from optimised and non-optimised OMP'
test_get_best_gene_first_iter_equality()


@pytest.mark.optimised
def test_get_best_gene_base_equality():
    from coppafish.omp.coefs import get_best_gene_base
    from coppafish.omp.coefs_optimised import get_best_gene_base as get_best_gene_base_jax

    rng = np.random.RandomState(98)
    n_rounds = 3
    n_channels = 4
    n_genes = 7
    # We test on one pixel because the jax code does a single pixel at a time
    residual_pixel_colors = rng.rand(n_rounds * n_channels)
    all_bled_codes = rng.rand(n_genes, n_rounds * n_channels)
    norm_shift = rng.rand()
    score_thresh = rng.rand() * 0.01
    inverse_var = rng.rand(n_rounds * n_channels)
    ignore_genes = np.asarray([[]], dtype=int)
    best_gene, pass_score_thresh = \
        get_best_gene_base(residual_pixel_colors, all_bled_codes, norm_shift, score_thresh, inverse_var, ignore_genes)
    best_gene_optimised, pass_score_thresh_optimised = \
        get_best_gene_base_jax(residual_pixel_colors, all_bled_codes, norm_shift, score_thresh, inverse_var, ignore_genes)
    assert best_gene == best_gene_optimised, 'Expected the same gene as the result'
    assert pass_score_thresh == pass_score_thresh_optimised, 'Expected the same boolean pass result'


@pytest.mark.optimised
def test_get_best_gene_equality():
    from coppafish.omp.coefs import get_best_gene
    from coppafish.omp.coefs_optimised import get_best_gene as get_best_gene_jax

    rng = np.random.RandomState(131)
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
    best_gene, pass_score_thresh, inverse_var = \
        get_best_gene(residual_pixel_colors, all_bled_codes, coefs, genes_added, norm_shift, score_thresh, alpha, 
                      background_genes, background_var)
    best_gene_optimised, pass_score_thresh_optimised, inverse_var_optimised = \
        get_best_gene_jax(residual_pixel_colors, all_bled_codes, coefs, genes_added, norm_shift, score_thresh, alpha, 
                      background_genes, background_var)
    assert np.allclose(best_gene, best_gene_optimised, atol=1e-4), 'Expected the same `best_genes` output'
    assert np.all(pass_score_thresh == pass_score_thresh_optimised), 'Expected the same `pass_score_thresh` output'
    assert np.allclose(inverse_var, inverse_var_optimised), 'Expected similar `inverse_var` output'


@pytest.mark.optimised
def test_get_all_coefs_equality():
    rng = np.random.RandomState(162)
    n_rounds = 6
    n_channels = 7
    n_genes = 8
    bled_codes = rng.rand(n_genes, n_rounds, n_channels)
    background_shift = rng.rand() * 0.001
    dp_shift = rng.rand() * 0.001
    dp_thresh = rng.rand() * 0.001
    alpha = rng.rand()
    beta = rng.rand()
    max_genes = 9
    for weight_coef_fit in [True, False]:
        n_pixels = 5
        pixel_colours = rng.rand(n_pixels, n_rounds, n_channels)
        from coppafish.omp.coefs import get_all_coefs
        gene_coefs, background_coefs = get_all_coefs(pixel_colours, bled_codes, background_shift, dp_shift, dp_thresh, 
                                                     alpha, beta, max_genes, weight_coef_fit)
        assert gene_coefs.shape == (n_pixels, n_genes)
        assert background_coefs.shape == (n_pixels, n_channels)
        from coppafish.omp.coefs_optimised import get_all_coefs
        gene_coefs_optimised, background_coefs_optimised = get_all_coefs(pixel_colours, bled_codes, background_shift, 
                                                                         dp_shift, dp_thresh, alpha, beta, max_genes, 
                                                                         weight_coef_fit)
        assert gene_coefs_optimised.shape == (n_pixels, n_genes)
        assert background_coefs_optimised.shape == (n_pixels, n_channels)
        assert np.allclose(gene_coefs, gene_coefs_optimised, atol=1e-4), 'Expected similar gene coefs'
        assert np.allclose(background_coefs, background_coefs_optimised, atol=1e-4), 'Expected similar background coefs'
