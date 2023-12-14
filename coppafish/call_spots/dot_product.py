from typing import Tuple
import numpy as np


def dot_product_score(spot_colours: np.ndarray, bled_codes: np.ndarray, weight_squared: np.ndarray = None,
                      norm_shift: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple dot product score assigning each spot to the gene with the highest score.

    Args:
        spot_colours (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`): spot colours.
        bled_codes (`[n_genes x (n_rounds * n_channels_use)] ndarray[float]`): normalised bled codes.
        weight_squared (`[n_spots x (n_rounds * n_channels_use)] ndarray[float]`, optional): array of weights. Default: 
            all ones.
        norm_shift (float, optional): added to the norm of each spot colour to avoid boosting weak spots too much. 
            Default: 0.

    Returns:
        - gene_no: np.ndarray of gene numbers [n_spots]
        - gene_score: np.ndarray of gene scores [n_spots]
        - gene_score_second: np.ndarray of second-best gene scores [n_spots]
        - `[n_spots x n_genes] ndarray[float]`: `score` such that `score[d, c]` gives dot product between 
            `spot_colours` vector `d` with `bled_codes` vector `c`.
    """
    n_spots, n_genes = spot_colours.shape[0], bled_codes.shape[0]
    n_rounds_channels_use = spot_colours.shape[1]
    # If no weighting is given, use equal weighting
    if weight_squared is None:
        weight_squared = np.ones((n_spots, n_rounds_channels_use))
        
    # Ensure bled_codes is normalised for each gene
    bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=1, keepdims=True)
    weight_squared = weight_squared / np.sum(weight_squared, axis=1)[:, None]
    spot_colours = spot_colours / (np.linalg.norm(spot_colours, axis=1)[:, None] + norm_shift)
    spot_colours = n_rounds_channels_use * spot_colours * weight_squared

    # Now we can obtain the dot product score for each spot and each gene
    all_score = spot_colours @ bled_codes.T
    gene_no = np.argmax(all_score, axis=1)
    all_score_sorted = np.sort(all_score, axis=1)
    gene_score = all_score_sorted[:, -1]
    gene_score_second = all_score_sorted[:, -2]

    return gene_no, gene_score, gene_score_second, all_score


def gene_prob_score(spot_colours: np.ndarray, bled_codes: np.ndarray, kappa: float = 2) -> np.ndarray:
    """
    Probability model says that for each spot in a particular round, the normalised fluorescence vector follows a
    Von-Mises Fisher distribution with mean equal to the normalised fluorescence for each dye and concentration
    parameter kappa. Then invert this to get prob(dye | fluorescence) and multiply across rounds to get
    prob(gene | spot_colours).
    
    Args:
        spot_colours (`(n_spots x n_rounds x n_channels_use) ndarray`): spot colours.
        bled_codes (`(n_genes x n_rounds x n_channels_use) ndarray`): normalised bled codes.
        kappa (float, optional), scaling factor for dot product score. Default: 2.
        
    Returns:
        probability: np.ndarray of gene probabilities [n_spots, n_genes]
    """
    n_spots, n_genes = spot_colours.shape[0], bled_codes.shape[0]
    # First, normalise spot_colours so that for each spot s and round r, norm(spot_colours[s, r, :]) = 1
    spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=2)[:, :, None]
    # Do the same for bled_codes
    bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=2)[:, :, None]
    # At this point, reshape spot_colours to be [n_spots, n_rounds * n_channels_use] and bled_codes to be
    # [n_genes, n_rounds * n_channels_use]
    spot_colours = spot_colours.reshape((n_spots, -1))
    bled_codes = bled_codes.reshape((n_genes, -1))

    # Now we can compute the dot products of each spot with each gene, producing a matrix of shape [n_spots, n_genes]
    dot_product = spot_colours @ bled_codes.T
    probability = np.exp(kappa * dot_product)
    # Now normalise so that each row sums to 1
    probability = probability / np.sum(probability, axis=1)[:, None]

    return probability
