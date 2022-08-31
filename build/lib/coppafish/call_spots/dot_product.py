from typing import Optional
import numpy as np
from .. import utils


def dot_product_score(spot_colors: np.ndarray, bled_codes: np.ndarray, norm_shift: float = 0,
                      weight_squared: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes `sum(W**2(s * b) / W**2)` where `s` is a `spot_color`, `b` is a `bled_code` and `W**2` is weight_squared
    for a particular `spot_color`. Sum is over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x (n_rounds x n_channels)]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        weight_squared: `float [n_spots x (n_rounds x n_channels)]`.
            squared weight to apply to each round/channel for each spot when computing dot product.
            If `None`, all rounds, channels treated equally.

    Returns:
        `float [n_spots x n_genes]`.
            `score` such that `score[d, c]` gives dot product between `spot_colors` vector `d`
            with `bled_codes` vector `c`.
    """
    n_spots = spot_colors.shape[0]
    n_genes, n_round_channels = bled_codes.shape
    if not utils.errors.check_shape(spot_colors[0], bled_codes[0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      (n_spots, n_round_channels))
    spot_norm_factor = np.linalg.norm(spot_colors, axis=1, keepdims=True)
    spot_norm_factor = spot_norm_factor + norm_shift
    spot_colors = spot_colors / spot_norm_factor

    gene_norm_factor = np.linalg.norm(bled_codes, axis=1, keepdims=True)
    gene_norm_factor[gene_norm_factor == 0] = 1  # so don't blow up if bled_code is all 0 for a gene.
    bled_codes = bled_codes / gene_norm_factor

    if weight_squared is not None:
        if not utils.errors.check_shape(weight_squared, spot_colors.shape):
            raise utils.errors.ShapeError('weight', weight_squared.shape,
                                          spot_colors.shape)
        spot_colors = spot_colors * weight_squared

    score = spot_colors @ bled_codes.transpose()

    if weight_squared is not None:
        score = score / np.expand_dims(np.sum(weight_squared, axis=1), 1)
        score = score * n_round_channels  # make maximum score 1 if all weight the same and dot product perfect.

    return score


def dot_product_score_no_weight(spot_colors: np.ndarray, bled_codes: np.ndarray, norm_shift: float = 0) -> np.ndarray:
    """
    Computes `sum((s * b))` where `s` is a `spot_color`, `b` is a `bled_code`.
    Sum is over all rounds and channels.

    Args:
        spot_colors: `float [n_spots x (n_rounds x n_channels)]`.
            Spot colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x (n_rounds x n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.

    Returns:
        `float [n_spots x n_genes]`.
            `score` such that `score[d, c]` gives dot product between `spot_colors` vector `d`
            with `bled_codes` vector `c`.
    """
    n_spots = spot_colors.shape[0]
    n_genes, n_round_channels = bled_codes.shape
    if not utils.errors.check_shape(spot_colors[0], bled_codes[0].shape):
        raise utils.errors.ShapeError('spot_colors', spot_colors.shape,
                                      (n_spots, n_round_channels))
    spot_norm_factor = np.linalg.norm(spot_colors, axis=1, keepdims=True)
    spot_norm_factor = spot_norm_factor + norm_shift
    spot_colors = spot_colors / spot_norm_factor

    gene_norm_factor = np.linalg.norm(bled_codes, axis=1, keepdims=True)
    gene_norm_factor[gene_norm_factor == 0] = 1  # so don't blow up if bled_code is all 0 for a gene.
    bled_codes = bled_codes / gene_norm_factor

    score = spot_colors @ bled_codes.transpose()
    return score
