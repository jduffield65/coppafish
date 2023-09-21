from coppafish.call_spots.base import get_non_duplicate, get_bled_codes, compute_gene_efficiency

import numpy as np


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
