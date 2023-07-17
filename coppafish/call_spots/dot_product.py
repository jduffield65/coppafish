from typing import Tuple
import numpy as np
from tqdm import tqdm


def dot_product_score(spot_colours: np.ndarray, bled_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple dot product score assigning each spot to the gene with the highest score.
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
    # TODO: Maybe do this normalisation with an extra term on the denominator
    spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=1)[:, None]
    bled_codes = bled_codes.reshape(n_genes, -1)

    all_score = spot_colours @ bled_codes.T
    gene_no = np.argmax(all_score, axis=1)
    all_score = np.sort(all_score, axis=1)
    gene_score = all_score[:, -1]
    gene_score_second = all_score[:, -2]

    return gene_no, gene_score, gene_score_second


def gene_prob_score(spot_colours: np.ndarray, bleed_matrix: np.ndarray, gene_codes: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
   Score assigning each spot to a gene. For each spot, we obtain n_rounds vectors of length n_channels_use. We then
   assign each of these vectors to a dye. This is more versatile than the dot product score as it allows for different
   strengths of dye in different rounds.
    Args:
        spot_colours: np.ndarray of spot colours [n_spots, n_rounds * n_channels_use]
        bleed_matrix: np.ndarray of bleed matrix [n_channels_use, n_dyes]
        gene_codes: np.ndarray of gene codes [n_genes]
    Returns:
        gene_probabilities: np.ndarray of gene probabilities [n_spots, n_genes]
        dye_strengths: np.ndarray of dye strengths [n_spots, n_rounds, n_dyes]
    """
    n_spots, n_rounds, n_genes, n_dyes = spot_colours.shape[0], spot_colours.shape[1], \
        gene_codes.shape[0], bleed_matrix.shape[1]
    # For each spot, we would like to obtain a dye strength for each dye in each round. So we define dye_strengths
    # as a matrix of shape [n_spots, n_rounds, n_dyes]
    dye_strengths = np.zeros((n_spots, n_rounds, n_dyes))
    for r in range(n_rounds):
        dye_strengths[:, r] = spot_colours[:, r] @ bleed_matrix

    # Now we can convert these dye strengths to probabilities
    dye_probabilities = np.abs(dye_strengths) / np.sum(np.abs(dye_strengths), axis=2)[:, :, None]
    # We can now obtain the probability of each gene in each spot
    gene_probabilities = np.ones((n_spots, n_genes))
    for g in tqdm(range(n_genes)):
        for s in range(n_spots):
            for r in range(n_rounds):
                gene_probabilities[s, g] *= dye_probabilities[s, r, gene_codes[g, r]]

    # Finally, normalise the gene probabilities so that the probabilities for each spot sum to 1
    gene_probabilities = gene_probabilities / np.sum(gene_probabilities, axis=1)[:, None]
    return gene_probabilities, dye_strengths
