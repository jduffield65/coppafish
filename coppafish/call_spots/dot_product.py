from typing import Tuple
import numpy as np


def compute_gene_scores(spot_colours: np.ndarray, bled_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score algorithm based on matching pursuit. For each spot, find the gene that matches best and remove it from the
    list of genes. Then find the next best match and so on. Then for each spot we will save an array of coefficients
    for each gene.
    Args:
        spot_colours: np.ndarray of spot colours [n_spots, n_rounds * n_channels_use]
        bled_codes: np.ndarray of normalised bled codes [n_genes, n_rounds, n_channels_use]
    Returns:
        gene_coefficients: np.ndarray of coefficients for each gene [n_spots, n_genes]
    """
    n_spots, n_genes = spot_colours.shape[0], bled_codes.shape[0]
    # First convert these matrices to vectors so that we can use the dot product
    spot_colours = spot_colours.reshape(n_spots, -1)
    bled_codes = bled_codes.reshape(n_genes, -1)
    gene_score = np.zeros(n_spots)
    gene_score_second = np.zeros(n_spots)
    gene_no = np.zeros(n_spots, dtype=int)

    for s in range(n_spots):
        spot_s_colour = spot_colours[s]
        spot_s_colour = spot_s_colour / np.linalg.norm(spot_s_colour)
        scores = spot_s_colour @ bled_codes.T
        gene_no[s] = np.argmax(scores)
        gene_score[s] = np.max(scores)
        gene_score_second[s] = np.max(np.delete(scores, gene_no[s]))

    return gene_no, gene_score, gene_score_second
