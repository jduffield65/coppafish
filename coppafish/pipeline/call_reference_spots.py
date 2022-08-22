from ..call_spots import get_dye_channel_intensity_guess, get_bleed_matrix, get_bled_codes, color_normalisation, \
    dot_product_score, get_spot_intensity, fit_background, get_gene_efficiency
import numpy as np
from ..setup.notebook import NotebookPage
from ..extract import scale
from ..spot_colors import get_spot_colors, all_pixel_yxz
from ..utils import round_any
from typing import Tuple
import warnings


def call_reference_spots(config: dict, nbp_file: NotebookPage, nbp_basic: NotebookPage, nbp_ref_spots: NotebookPage,
                         hist_values: np.ndarray, hist_counts: np.ndarray,
                         transform: np.ndarray, overwrite_ref_spots: bool = False) -> Tuple[NotebookPage, NotebookPage]:
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
        hist_values: `int [n_pixel_values]`.
            All possible pixel values in saved tiff images i.e. `n_pixel_values` is approximately
            `np.iinfo(np.uint16).max` because tiffs saved as `uint16` images.
            This is saved in the extract notebook page i.e. `nb.extract.hist_values`.
        hist_counts: `int [n_pixel_values x n_rounds x n_channels]`.
            `hist_counts[i, r, c]` is the number of pixels across all tiles in round `r`, channel `c`
            which had the value `hist_values[i]`.
            This is saved in extract notebook page i.e. `nb.extract.hist_counts`.
        transform: `float [n_tiles x n_rounds x n_channels x 4 x 3]`.
            `transform[t, r, c]` is the affine transform to get from tile `t`, `ref_round`, `ref_channel` to
            tile `t`, round `r`, channel `c`.
            This is saved in the register notebook page i.e. `nb.register.transform`.
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
        warnings.warn("\noverwrite_ref_spots = True so will overwrite:\ngene_no, score, score_diff, intensity"
                      "\nin nbp_ref_spots.")
    else:
        # Raise error if data in nbp_ref_spots already exists that will be overwritted in this function.
        error_message = ""
        for var in ['gene_no', 'score', 'score_diff', 'intensity']:
            if hasattr(nbp_ref_spots, var) and nbp_ref_spots.__getattribute__(var) is not None:
                error_message += f"\nnbp_ref_spots.{var} is not None but this function will overwrite {var}." \
                                 f"\nRun with overwrite_ref_spots = True to get past this error."
        if len(error_message) > 0:
            raise ValueError(error_message)

    nbp_ref_spots.finalized = False  # So we can add and delete ref_spots page variables
    # delete all variables in ref_spots set to None so can add them later.
    for var in ['gene_no', 'score', 'score_diff', 'intensity']:
        if hasattr(nbp_ref_spots, var):
            nbp_ref_spots.__delattr__(var)
    nbp = NotebookPage("call_spots")

    # get color norm factor
    rc_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels)
    hist_counts_use = np.moveaxis(np.moveaxis(hist_counts, 0, -1)[rc_ind], -1, 0)
    color_norm_factor = np.ones((nbp_basic.n_rounds, nbp_basic.n_channels)) * np.nan
    color_norm_factor[rc_ind] = color_normalisation(hist_values, hist_counts_use, config['color_norm_intensities'],
                                                    config['color_norm_probs'], config['bleed_matrix_method'])

    # get initial bleed matrix
    initial_raw_bleed_matrix = np.ones((nbp_basic.n_rounds, nbp_basic.n_channels, nbp_basic.n_dyes)) * np.nan
    rcd_ind = np.ix_(nbp_basic.use_rounds, nbp_basic.use_channels, nbp_basic.use_dyes)
    if nbp_basic.dye_names is not None:
        # if specify dyes, will initialize bleed matrix using prior data
        dye_names_use = np.array(nbp_basic.dye_names)[nbp_basic.use_dyes]
        camera_use = np.array(nbp_basic.channel_camera)[nbp_basic.use_channels]
        laser_use = np.array(nbp_basic.channel_laser)[nbp_basic.use_channels]
        initial_raw_bleed_matrix[rcd_ind] = get_dye_channel_intensity_guess(nbp_file.dye_camera_laser,
                                                                            dye_names_use, camera_use,
                                                                            laser_use).transpose()
        initial_bleed_matrix = initial_raw_bleed_matrix / np.expand_dims(color_norm_factor, 2)
    else:
        if nbp_basic.n_dyes != nbp_basic.n_channels:
            raise ValueError(f"'dye_names' were not specified so expect each dye to correspond to a different channel."
                             f"\nBut n_channels={nbp_basic.n_channels} and n_dyes={nbp_basic.n_dyes}")
        if nbp_basic.use_channels != nbp_basic.use_dyes:
            raise ValueError(f"'dye_names' were not specified so expect each dye to correspond to a different channel."
                             f"\nBleed matrix computation requires use_channels and use_dyes to be the same to work."
                             f"\nBut use_channels={nbp_basic.use_channels} and use_dyes={nbp_basic.use_dyes}")
        initial_bleed_matrix = initial_raw_bleed_matrix.copy()
        initial_bleed_matrix[rcd_ind] = np.tile(np.expand_dims(np.eye(nbp_basic.n_channels), 0),
                                                (nbp_basic.n_rounds, 1, 1))[rcd_ind]

    # Get norm_shift and intensity_thresh from middle tile/ z-plane average intensity
    # This is because these variables are all a small fraction of a spot_color L2 norm in one round.
    # Hence, use average pixel as example of low intensity spot.
    # get central tile
    nbp.norm_shift_tile = scale.central_tile(nbp_basic.tilepos_yx, nbp_basic.use_tiles)
    if nbp_basic.is_3d:
        nbp.norm_shift_z = int(np.floor(nbp_basic.nz / 2))  # central z-plane to get info from.
    else:
        nbp.norm_shift_z = 0
    pixel_colors = get_spot_colors(all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, nbp.norm_shift_z),
                                   nbp.norm_shift_tile, transform, nbp_file, nbp_basic, return_in_bounds=True)[0]
    pixel_intensity = get_spot_intensity(np.abs(pixel_colors) / color_norm_factor[rc_ind])
    nbp.abs_intensity_percentile = np.percentile(pixel_intensity, np.arange(1, 101))
    if config['background_weight_shift'] is None:
        # Set to median absolute pixel intensity
        config['background_weight_shift'] = float(round_any(nbp.abs_intensity_percentile[50],
                                                            config['norm_shift_precision'], 'ceil'))
    median_round_l2_norm = np.median(np.linalg.norm(pixel_colors / color_norm_factor[rc_ind], axis=2))
    if config['dp_norm_shift'] is None:
        config['dp_norm_shift'] = float(round_any(median_round_l2_norm, config['norm_shift_precision']))
    # intensity thresh is just a very low threshold, would basically be the same if set to 0
    # but found it to be slightly better on ground truth
    pixel_intensity = get_spot_intensity(pixel_colors / color_norm_factor[rc_ind])
    if config['gene_efficiency_intensity_thresh'] is None:
        config['gene_efficiency_intensity_thresh'] = \
            float(round_any(np.percentile(pixel_intensity, config['gene_efficiency_intensity_thresh_percentile']),
                            config['gene_efficiency_intensity_thresh_precision']))
    nbp.dp_norm_shift = float(np.clip(config['dp_norm_shift'], config['norm_shift_min'], config['norm_shift_max']))
    nbp.background_weight_shift = float(np.clip(config['background_weight_shift'],
                                                config['norm_shift_min'], config['norm_shift_max']))
    nbp.gene_efficiency_intensity_thresh = \
        float(np.clip(config['gene_efficiency_intensity_thresh'],
                      config['gene_efficiency_intensity_thresh_min'],
                      config['gene_efficiency_intensity_thresh_max']))

    # get bleed matrix
    spot_colors_use = np.moveaxis(np.moveaxis(nbp_ref_spots.colors, 0, -1)[rc_ind], -1, 0) / color_norm_factor[rc_ind]
    nbp_ref_spots.intensity = np.asarray(get_spot_intensity(spot_colors_use).astype(np.float32))
    # Remove background first
    background_coef = np.ones((spot_colors_use.shape[0], nbp_basic.n_channels)) * np.nan
    background_codes = np.ones((nbp_basic.n_channels, nbp_basic.n_rounds, nbp_basic.n_channels)) * np.nan
    crc_ind = np.ix_(nbp_basic.use_channels, nbp_basic.use_rounds, nbp_basic.use_channels)
    spot_colors_use, background_coef[:, nbp_basic.use_channels], background_codes[crc_ind] = \
        fit_background(spot_colors_use, nbp.background_weight_shift)
    spot_colors_use = np.asarray(spot_colors_use)  # in case using jax
    bleed_matrix = initial_raw_bleed_matrix.copy()
    bleed_matrix[rcd_ind] = get_bleed_matrix(spot_colors_use[nbp_ref_spots.isolated], initial_bleed_matrix[rcd_ind],
                                             config['bleed_matrix_method'], config['bleed_matrix_score_thresh'],
                                             config['bleed_matrix_min_cluster_size'], config['bleed_matrix_n_iter'],
                                             config['bleed_matrix_anneal'])

    # get gene codes
    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str)).transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))])
    # bled_codes[g,r,c] returned below is nan where r/c/gene_codes[g,r] outside use_rounds/channels/dyes
    bled_codes = get_bled_codes(gene_codes, bleed_matrix)

    # get bled_codes_use with no nan values and L2 norm=1
    bled_codes_use = np.moveaxis(np.moveaxis(bled_codes, 0, -1)[rc_ind], -1, 0)
    bled_codes_use[np.isnan(bled_codes_use)] = 0  # set all round vectors where dye is not in use_dyes to 0.
    # Give all bled codes an L2 norm of 1 across use_rounds and use_channels
    norm_factor = np.expand_dims(np.linalg.norm(bled_codes_use, axis=(1, 2)), (1, 2))
    norm_factor[norm_factor == 0] = 1  # For genes with no dye in use_dye, this avoids blow up on next line
    bled_codes_use = bled_codes_use / norm_factor

    # bled_codes[g,r,c] so nan when r/c outside use_rounds/channels and 0 when gene_codes[g,r] outside use_dyes
    n_genes = bled_codes_use.shape[0]
    bled_codes = np.ones((nbp_basic.n_rounds, nbp_basic.n_channels, n_genes)) * np.nan
    bled_codes[rc_ind] = np.moveaxis(bled_codes_use, 0, -1)
    bled_codes = np.moveaxis(bled_codes, -1, 0)

    nbp.gene_names = gene_names
    nbp.gene_codes = gene_codes
    nbp.color_norm_factor = color_norm_factor
    nbp.initial_raw_bleed_matrix = initial_raw_bleed_matrix
    nbp.initial_bleed_matrix = initial_bleed_matrix
    nbp.bleed_matrix = bleed_matrix
    nbp.bled_codes = bled_codes
    nbp.background_codes = background_codes

    # shift in config file is just for one round.
    n_spots, n_rounds_use, n_channels_use = spot_colors_use.shape
    dp_norm_shift = nbp.dp_norm_shift * np.sqrt(n_rounds_use)

    # Down-weight round/channels with high background when compute dot product
    alpha = config['alpha']
    beta = config['beta']
    background_codes = background_codes[crc_ind].reshape(n_channels_use, -1)
    background_var = background_coef[:, nbp_basic.use_channels]**2 @ background_codes**2 * alpha + beta ** 2

    # find spot assignments to genes and gene efficiency
    n_iter = config['gene_efficiency_n_iter'] + 1
    pass_intensity_thresh = nbp_ref_spots.intensity > nbp.gene_efficiency_intensity_thresh
    use_ge_last = np.zeros(n_spots).astype(bool)
    bled_codes_ge_use = bled_codes_use.copy()
    for i in range(n_iter):
        scores = np.asarray(dot_product_score(spot_colors_use.reshape(n_spots, -1),
                                              bled_codes_ge_use.reshape(n_genes, -1), dp_norm_shift, 1/background_var))
        spot_gene_no = np.argmax(scores, 1)
        spot_score = scores[np.arange(np.shape(scores)[0]), spot_gene_no]
        pass_score_thresh = spot_score > config['gene_efficiency_score_thresh']

        sort_gene_inds = np.argsort(scores, axis=1)
        gene_no_second_best = sort_gene_inds[:, -2]
        score_second_best = scores[np.arange(np.shape(scores)[0]), gene_no_second_best]
        spot_score_diff = spot_score - score_second_best
        pass_score_diff_thresh = spot_score_diff > config['gene_efficiency_score_diff_thresh']
        # only use isolated spots which pass strict thresholding to compute gene_efficiencies
        use_ge = np.array([nbp_ref_spots.isolated, pass_intensity_thresh, pass_score_thresh,
                           pass_score_diff_thresh]).all(axis=0)
        # nan_to_num line below converts nan in bleed_matrix to 0.
        # This basically just says that for dyes not in use_dyes, we expect intensity to be 0.
        gene_efficiency_use = get_gene_efficiency(spot_colors_use[use_ge], spot_gene_no[use_ge],
                                                  gene_codes[:, nbp_basic.use_rounds],
                                                  np.nan_to_num(bleed_matrix[rc_ind]),
                                                  config['gene_efficiency_min_spots'],
                                                  config['gene_efficiency_max'],
                                                  config['gene_efficiency_min'],
                                                  config['gene_efficiency_min_factor'])

        # get new bled codes using gene efficiency with L2 norm = 1.
        multiplier_ge = np.tile(np.expand_dims(gene_efficiency_use, 2), [1, 1, n_channels_use])
        bled_codes_ge_use = bled_codes_use * multiplier_ge
        norm_factor = np.expand_dims(np.linalg.norm(bled_codes_ge_use, axis=(1, 2)), (1, 2))
        norm_factor[norm_factor == 0] = 1  # For genes with no dye in use_dye, this avoids blow up on next line
        bled_codes_ge_use = bled_codes_ge_use / norm_factor

        if np.sum(use_ge != use_ge_last) < 3:
            # if less than 3 spots different in spots used for ge computation, end.
            break
        use_ge_last = use_ge.copy()

    if config['gene_efficiency_n_iter'] > 0:
        # Compute score with final gene efficiency
        scores = np.asarray(dot_product_score(spot_colors_use.reshape(n_spots, -1),
                                              bled_codes_ge_use.reshape(n_genes, -1), dp_norm_shift, 1/background_var))
        spot_gene_no = np.argmax(scores, 1)
        spot_score = scores[np.arange(np.shape(scores)[0]), spot_gene_no]
    else:
        bled_codes_ge_use = bled_codes_use.copy()

    # save score using the latest gene efficiency and diff to second best gene
    nbp_ref_spots.score = spot_score.astype(np.float32)
    nbp_ref_spots.gene_no = spot_gene_no.astype(np.int16)
    sort_gene_inds = np.argsort(scores, axis=1)
    gene_no_second_best = sort_gene_inds[:, -2]
    score_second_best = scores[np.arange(np.shape(scores)[0]), gene_no_second_best]
    nbp_ref_spots.score_diff = (nbp_ref_spots.score - score_second_best).astype(np.float16)

    # save gene_efficiency[g,r] with nan when r outside use_rounds and 1 when gene_codes[g,r] outside use_dyes.
    gene_efficiency = np.ones((n_genes, nbp_basic.n_rounds)) * np.nan
    gene_efficiency[:, nbp_basic.use_rounds] = gene_efficiency_use
    nbp.gene_efficiency = gene_efficiency

    # bled_codes_ge[g,r,c] so nan when r/c outside use_rounds/channels and 0 when gene_codes[g,r] outside use_dyes
    bled_codes_ge = np.ones((nbp_basic.n_rounds, nbp_basic.n_channels, n_genes)) * np.nan
    bled_codes_ge[rc_ind] = np.moveaxis(bled_codes_ge_use, 0, -1)
    bled_codes_ge = np.moveaxis(bled_codes_ge, -1, 0)
    nbp.bled_codes_ge = bled_codes_ge

    ge_fail_genes = np.where(np.min(gene_efficiency_use,axis=1) == 1)[0]
    n_fail_ge = len(ge_fail_genes)
    if n_fail_ge > 0:
        fail_genes_str = [str(ge_fail_genes[i]) + ': ' + gene_names[ge_fail_genes][i] for i in range(n_fail_ge)]
        fail_genes_str = '\n'.join(fail_genes_str)
        warnings.warn(f"\nGene Efficiency could not be calculated for {n_fail_ge}/{n_genes} "
                      f"genes:\n{fail_genes_str}")
    nbp_ref_spots.finalized = True
    return nbp, nbp_ref_spots
