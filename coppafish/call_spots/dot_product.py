from typing import Tuple
import numpy as np


def dot_product_score(spot_colours: np.ndarray, bled_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score algorithm based on matching pursuit. For each spot, find the gene that matches best and remove it from the
    list of genes. Then find the next best match and so on. Then for each spot we will save an array of coefficients
    for each gene.
    Args:
        spot_colours: np.ndarray of spot colours [n_spots, n_rounds * n_channels_use]
        bled_codes: np.ndarray of normalised bled codes [n_genes, n_rounds, n_channels_use]
    Returns:
        gene_no: np.ndarray of gene numbers [n_spots]
        gene_score: np.ndarray of gene scores [n_spots]
        gene_score_second: np.ndarray of second-best gene scores [n_spots]
    """
    n_spots, n_genes = spot_colours.shape[0], bled_codes.shape[0]
    # First convert these matrices to vectors so that we can use the dot product
    spot_colours = spot_colours.reshape(n_spots, -1)
    # Normalise spot colours
    spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=1)[:, None]
    bled_codes = bled_codes.reshape(n_genes, -1)

    all_score = spot_colours @ bled_codes.T
    gene_no = np.argmax(all_score, axis=1)
    all_score = np.sort(all_score, axis=1)
    gene_score = all_score[:, -1]
    gene_score_second = all_score[:, -2]

    return gene_no, gene_score, gene_score_second
