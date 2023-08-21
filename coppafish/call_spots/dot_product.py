from typing import Tuple
import numpy as np
from tqdm import tqdm


def dot_product_score(spot_colours: np.ndarray, bled_codes: np.ndarray, weight_squared: np.ndarray = None,
                      norm_shift: float = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple dot product score assigning each spot to the gene with the highest score.
    Args:
        spot_colours: np.ndarray of spot colours [n_spots, n_rounds, n_channels_use]
        bled_codes: np.ndarray of normalised bled codes [n_genes, n_rounds, n_channels_use]
        weight_squared: np.ndarray of weights [n_rounds, n_channels_use]
        norm_shift: float to add to the norm of each spot colour to avoid boosting weak spots too much
    Returns:
        gene_no: np.ndarray of gene numbers [n_spots]
        gene_score: np.ndarray of gene scores [n_spots]
        gene_score_second: np.ndarray of second-best gene scores [n_spots]
    """
    # If no weighting is given, use equal weighting
    if weight_squared is None:
        weight_squared = np.ones(spot_colours.shape[1:])

    # Flatten the weight matrix
    weight_squared = weight_squared.reshape(-1)
    n_spots, n_genes = spot_colours.shape[0], bled_codes.shape[0]
    # First convert these matrices to vectors so that we can use the dot product.
    spot_colours = spot_colours.reshape(n_spots, -1)
    spot_colours = spot_colours / (np.linalg.norm(spot_colours, axis=1)[:, None] + norm_shift)
    spot_colours = spot_colours * weight_squared
    bled_codes = bled_codes.reshape(n_genes, -1)

    # Now we can obtain the dot product score for each spot and each gene
    all_score = spot_colours @ bled_codes.T
    gene_no = np.argmax(all_score, axis=1)
    all_score = np.sort(all_score, axis=1)
    gene_score = all_score[:, -1]
    gene_score_second = all_score[:, -2]

    return gene_no, gene_score, gene_score_second


# def dot_product_score(spot_colours: np.ndarray, bled_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Simple dot product score assigning each spot to the gene with the highest score.
#     Args:
#         spot_colours: np.ndarray of spot colours [n_spots, n_rounds, n_channels_use]
#         bled_codes: np.ndarray of normalised bled codes [n_genes, n_rounds, n_channels_use]
#     Returns:
#         gene_no: np.ndarray of gene numbers [n_spots]
#         gene_score: np.ndarray of gene scores [n_spots]
#         gene_score_second: np.ndarray of second-best gene scores [n_spots]
#     """
#     n_spots, n_genes = spot_colours.shape[0], bled_codes.shape[0]
#     # normalise each spot colour matrix so that for each spot s and round r, norm(spot_colours[s, r, :]) = 1
#     spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=2)[:, :, None]
#     # Do the same for bled_codes
#     bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=2)[:, :, None]
#     # At this point, reshape spot_colours to be [n_spots, n_rounds * n_channels_use] and bled_codes to be
#     # [n_genes, n_rounds * n_channels_use]
#     spot_colours = spot_colours.reshape((n_spots, -1))
#     bled_codes = bled_codes.reshape(n_genes, -1)
#     # earlier normalisation was to equalise across rounds, now ensure each spot has norm 1 and each gene has norm 1
#     spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=1)[:, None]
#     bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=1)[:, None]
#
#     # Now we can obtain the dot product score for each spot and each gene
#     all_score = spot_colours @ bled_codes.T
#     gene_no = np.argmax(all_score, axis=1)
#     all_score = np.sort(all_score, axis=1)
#     gene_score = all_score[:, -1]
#     gene_score_second = all_score[:, -2]
#
#     return gene_no, gene_score, gene_score_second


def gene_prob_score(spot_colours: np.ndarray, bled_codes: np.ndarray, kappa: float = 2) -> np.ndarray:
    """
    Probability model says that for each spot in a particular round, the normalised fluorescence vector follows a
    Von-Mises Fisher distribution with mean equal to the normalised fluorescence for each dye and concentration
    parameter kappa. Then invert this to get prob(dye | fluorescence) and multiply across rounds to get
    prob(gene | spot_colours).
    Args:
        spot_colours: np.ndarray of spot colours [n_spots, n_rounds, n_channels_use]
        bled_codes: np.ndarray of normalised bled codes [n_genes, n_rounds, n_channels_use]
        kappa: float, scaling factor for dot product score
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
