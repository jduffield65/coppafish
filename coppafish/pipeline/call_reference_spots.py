import os
import numpy as np
from typing import Tuple
import warnings

from ..setup.notebook import NotebookPage
from .. import call_spots
from .. import spot_colors
from .. import utils


def call_reference_spots(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage,
                         nbp_ref_spots: NotebookPage, nbp_extract: NotebookPage, transform: np.ndarray,
                         overwrite_ref_spots: bool = False) -> Tuple[NotebookPage, NotebookPage]:
    """
    This produces the bleed matrix and expected code for each gene as well as producing a gene assignment based on a
    simple dot product for spots found on the reference round.

    Returns the `call_spots` notebook page and adds the following variables to the `ref_spots` page:
    `gene_no`, `score`, `score_diff`, `intensity`.

    See `'call_spots'` and `'ref_spots'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        config: Dictionary obtained from `'call_spots'` section of config file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page
        nbp_ref_spots: `ref_spots` notebook page containing all variables produced in `pipeline/reference_spots.py` i.e.
            `local_yxz`, `isolated`, `tile`, `colors`.
            `gene_no`, `score`, `score_diff`, `intensity` should all be `None` to add them here, unless
            `overwrite_ref_spots == True`.
        transform: float [n_tiles x n_rounds x n_channels x 4 x 3] affine transform for each tile, round and channel
        overwrite_ref_spots: If `True`, the variables:
            * `gene_no`
            * `score`
            * `score_diff`
            * `intensity`

            in `nbp_ref_spots` will be overwritten if they exist. If this is `False`, they will only be overwritten
            if they are all set to `None`, otherwise an error will occur.

    Returns:
        `NotebookPage[call_spots]` - Page contains bleed matrix and expected code for each gene.
        `NotebookPage[ref_spots]` - Page contains gene assignments and info for spots found on reference round.
            Parameters added are: intensity, score, gene_no, score_diff
    """
    if overwrite_ref_spots:
        warnings.warn("\noverwrite_ref_spots = True so will overwrite:\ngene_no, gene_score, score_diff, intensity,"
                      "\nbackground_strength in nbp_ref_spots.")
    else:
        # Raise error if data in nbp_ref_spots already exists that will be overwritten in this function.
        error_message = ""
        for var in ['gene_no', 'gene_score', 'score_diff', 'intensity', 'background_strength', 'gene_probs',
                    'dye_strengths']:
            if hasattr(nbp_ref_spots, var) and nbp_ref_spots.__getattribute__(var) is not None:
                error_message += f"\nnbp_ref_spots.{var} is not None but this function will overwrite {var}." \
                                 f"\nRun with overwrite_ref_spots = True to get past this error."
        if len(error_message) > 0:
            raise ValueError(error_message)

    nbp_ref_spots.finalized = False  # So we can add and delete ref_spots page variables
    # delete all variables in ref_spots set to None so can add them later.
    for var in ['gene_no', 'score', 'score_diff', 'intensity', 'background_strength',  'gene_probs', 'dye_strengths']:
        if hasattr(nbp_ref_spots, var):
            nbp_ref_spots.__delattr__(var)
    nbp = NotebookPage("call_spots")

    # 0. Initialise frequently used variables
    n_rounds, use_channels = nbp_basic.n_rounds, nbp_basic.use_channels
    spot_colours = nbp_ref_spots.colors[:, :, nbp_basic.use_channels]

    # 1. Remove background from spots and normalise channels and rounds
    # Find middle tile to calculate intensity threshold
    spot_colours_background_removed, background_noise = spot_colors.remove_background(spot_colours=spot_colours.copy())
    median_tile = int(np.median(nbp_basic.use_tiles))
    dist = np.linalg.norm(nbp_basic.tilepos_yx - nbp_basic.tilepos_yx[median_tile], axis=1)[nbp_basic.use_tiles]
    central_tile = nbp_basic.use_tiles[np.argmin(dist)]
    pixel_colors = spot_colors.get_spot_colors(spot_colors.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, 
                                                                         nbp_basic.nz // 2), 
                                               central_tile, transform, nbp_file, nbp_basic, nbp_extract, 
                                               bg_scale=nbp_extract.bg_scale, return_in_bounds=True)[0]
    # normalise pixel colours by round and channel and then remove background
    # colour_norm_factor = normalise_rc(pixel_colors.astype(float), spot_colours_background_removed)
    colour_norm_factor = np.percentile(abs(pixel_colors), 99, axis=0)
    spot_colours = spot_colours_background_removed / colour_norm_factor[None]

    # save pixel intensity and delete pixel_colors to save memory
    pixel_intensity = call_spots.get_spot_intensity(np.abs(pixel_colors / colour_norm_factor[None]))
    nbp.abs_intensity_percentile = np.percentile(pixel_intensity, np.arange(1, 101))
    del pixel_colors, pixel_intensity

    # 2. Bleed matrix calculation and bled codes
    # 2.1 Calculate bleed matrix, this just involves normalising the bleed matrix template with the colour norm factor,
    # although if we are not creating a new bleed matrix for all rounds, we need to normalise the bleed matrix template
    # with a median of the colour norm factor across rounds.
    if nbp_file.initial_bleed_matrix is None:
        expected_dye_names = ['ATTO425', 'AF488', 'DY520XL', 'AF532', 'AF594', 'AF647', 'AF750']
        assert nbp_basic.dye_names == expected_dye_names, \
            f'To use the default bleed matrix, dye names must be given in the order {expected_dye_names}, but got ' \
                + f'{nbp_basic.dye_names}.'
        default_bleed_matrix_filepath = os.path.join(os.getcwd(), 'coppafish/setup/default_bleed.npy')
        initial_bleed_matrix = np.load(default_bleed_matrix_filepath).copy()
    if nbp_file.initial_bleed_matrix is not None:
        # Use an initial bleed matrix given by the user
        initial_bleed_matrix = np.load(nbp_file.initial_bleed_matrix)
    expected_shape = (len(nbp_basic.use_channels), len(nbp_basic.dye_names))
    assert initial_bleed_matrix.shape == expected_shape, \
        f'Initial bleed matrix at {nbp_file.initial_bleed_matrix} has shape {initial_bleed_matrix.shape}, ' \
            + f'expected {expected_shape}.'
    # normalise bleed matrix across channels, then once again across dyes so each column has norm 1
    bleed_norm = np.median(colour_norm_factor, axis=0)
    # Want to divide each row by bleed_norm, so reshape bleed_norm to be n_channels x n_rounds
    bleed_norm = np.repeat(bleed_norm[:, np.newaxis], n_rounds, axis=1)
    initial_bleed_matrix = initial_bleed_matrix / bleed_norm
    # now normalise each column (dye) to have norm 1
    bleed_matrix = initial_bleed_matrix / np.linalg.norm(initial_bleed_matrix, axis=0)
    # Repeat bleed n_rounds times along a new 0th axis
    bleed_matrix = np.repeat(bleed_matrix[np.newaxis, :, :], n_rounds, axis=0)
    intensity = call_spots.get_spot_intensity(spot_colors=spot_colours)

    # 2.2 Calculate bled codes and gene probabilities
    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
    n_genes = len(gene_names)
    ge_initial = np.ones((n_genes, n_rounds))
    bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix, gene_efficiency=ge_initial)
    gene_prob = call_spots.gene_prob_score(spot_colours=spot_colours, bled_codes=bled_codes)
    gene_no = np.argmax(gene_prob, axis=1)
    gene_score = np.max(gene_prob, axis=1)

    # 3. Gene efficiency calculation.
    # GE calculation is done iteratively in a similar way to scaled k-means clustering. We start with our initial
    # score distribution and bled codes and then these 2 parameters are iteratively updated until convergence.
    # 3.1 Calculate gene efficiency
    ge_intensity_thresh = nbp.abs_intensity_percentile[int(config['gene_efficiency_intensity_thresh_percentile'])]
    gene_efficiency, use_ge, _ = call_spots.compute_gene_efficiency(spot_colours=spot_colours, bled_codes=bled_codes, 
                                                                    gene_no=gene_no, gene_score=gene_score, 
                                                                    gene_codes=gene_codes, intensity=intensity, 
                                                                    score_threshold=0.6, 
                                                                    intensity_threshold=ge_intensity_thresh)
    # 3.2 Update bled codes
    bled_codes = call_spots.get_bled_codes(gene_codes=gene_codes, bleed_matrix=bleed_matrix, 
                                           gene_efficiency=gene_efficiency)

    # 3.3 Update gene coefficients
    n_spots = spot_colours.shape[0]
    n_genes = bled_codes.shape[0]
    gene_no, gene_score, gene_score_second \
        = call_spots.dot_product_score(spot_colours=spot_colours.reshape((n_spots, -1)), 
                                       bled_codes=bled_codes.reshape((n_genes, -1)))[:3]

    # save overwritable variables in nbp_ref_spots
    nbp_ref_spots.gene_no = gene_no
    nbp_ref_spots.score = gene_score
    nbp_ref_spots.score_diff = gene_score - gene_score_second
    nbp_ref_spots.intensity = np.median(np.max(spot_colours, axis=2), axis=1).astype(np.float32)
    nbp_ref_spots.background_strength = background_noise
    nbp_ref_spots.gene_probs = gene_prob
    # nbp_ref_spots.dye_strengths = dye_strength
    nbp_ref_spots.finalized = True

    # Save variables in nbp
    nbp.use_ge = np.asarray(use_ge)
    nbp.gene_names = gene_names
    nbp.gene_codes = gene_codes
    # Now expand variables to have n_channels channels instead of n_channels_use channels. For some variables, we
    # also need to swap axes as the expand channels function assumes the last axis is the channel axis.
    nbp.color_norm_factor = utils.base.expand_channels(colour_norm_factor, use_channels, nbp_basic.n_channels)
    nbp.initial_bleed_matrix = utils.base.expand_channels(initial_bleed_matrix.T, use_channels, nbp_basic.n_channels).T
    nbp.bleed_matrix = utils.base.expand_channels(bleed_matrix.swapaxes(1, 2), use_channels, 
                                                  nbp_basic.n_channels).swapaxes(1, 2)
    # bled_codes_ge is what we have been calling bled_codes. Haven't kept a record of bled_codes before gene efficiency
    # was applied, so we need to recalculate it here.
    nbp.bled_codes_ge = utils.base.expand_channels(bled_codes, use_channels, nbp_basic.n_channels)
    nbp.bled_codes = utils.base.expand_channels(call_spots.get_bled_codes(gene_codes=gene_codes, 
                                                                          bleed_matrix=bleed_matrix, 
                                                                          gene_efficiency=ge_initial), 
                                                use_channels, nbp_basic.n_channels)
    nbp.gene_efficiency = gene_efficiency
    nbp.gene_efficiency_intensity_thresh = float(np.round(ge_intensity_thresh, 2))

    return nbp, nbp_ref_spots
