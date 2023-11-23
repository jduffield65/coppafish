import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Union, Any, Optional
from tqdm import tqdm
import scipy
import psutil
import multiprocessing

from .. import call_spots
from .. import utils
from .. import spot_colors
from ..call_spots import dot_product
from ..setup import NotebookPage


def fit_coefs(bled_codes: npt.NDArray, pixel_colors: npt.NDArray, genes: npt.NDArray) \
    -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Old method before Jax.
    This finds the least squared solution for how the `n_genes` `bled_codes` can best explain each `pixel_color`.
    Can also find weighted least squared solution if `weight` provided.

    Args:
        bled_codes: `float [(n_rounds * n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds * n_channels) x n_pixels]` if `n_genes==1`
            otherwise  `float [(n_rounds * n_channels)]`.
            Flattened then transposed pixel colors which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.

    Returns:
        - residual - `float [n_pixels x (n_rounds * n_channels)]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coef.
        - coefs - `float [n_pixels x n_genes_add]` if n_genes == 1 otherwise `float [n_genes]` if n_pixels == 1.
            coefficient found through least squares fitting for each gene.

    """
    n_pixels = pixel_colors.shape[1]
    residual = np.zeros((n_pixels, pixel_colors.shape[0]))
    coefs = np.zeros_like(genes, dtype=float)
    #TODO: Eliminate for loop with numpy magic
    for p in range(n_pixels):
        coefs[p] = np.linalg.lstsq(bled_codes[:, genes[p]], pixel_colors[:, p], rcond=None)[0]
        residual[p] = pixel_colors[:, p] - bled_codes[:, genes[p]] @ coefs[p]
    
    return residual, coefs


def fit_coefs_weight(bled_codes: np.ndarray, pixel_colors: np.ndarray, genes: np.ndarray,
                     weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This finds the weighted least squared solution for how the `n_genes_add` `bled_codes` indicated by `genes[s]`
    can best explain `pixel_colors[:, s]` for each pixel s. The `weight` indicates which rounds/channels should
    have more influence when finding the coefficients of each gene.

    Args:
        bled_codes: `float [(n_rounds * n_channels) x n_genes]`.
            Flattened then transposed bled codes which usually has the shape `[n_genes x n_rounds x n_channels]`.
        pixel_colors: `float [(n_rounds * n_channels) x n_pixels]`.
            Flattened then transposed `pixel_colors` which usually has the shape `[n_pixels x n_rounds x n_channels]`.
        genes: `int [n_pixels x n_genes_add]`.
            Indices of codes in bled_codes to find coefficients for which best explain each pixel_color.
        weight: `float [n_pixels x (n_rounds * n_channels)]`.
            `weight[s, i]` is the weight to be applied to round_channel `i` when computing coefficient of each
            `bled_code` for pixel `s`.

    Returns:
        - residual - `float32 [n_pixels x (n_rounds * n_channels)]`.
            Residual pixel_colors after removing bled_codes with coefficients specified by coefs.
        - coefs - `float32 [n_pixels x n_genes_add]`.
            Coefficients found through least squares fitting for each gene.
    """
    n_pixels, n_genes_add = genes.shape
    n_rounds_channels = bled_codes.shape[0]

    #TODO: Eliminate for loop with numpy magic
    residuals = np.zeros((n_pixels, n_rounds_channels), dtype=np.float32)
    coefs = np.zeros((n_pixels, n_genes_add), dtype=np.float32)
    for p in range(n_pixels):
        pixel_colour = pixel_colors[:,p]
        gene = genes[p]
        w = weight[p]
        coefs[p] = np.linalg.lstsq(bled_codes[:, gene] * w[:, np.newaxis], pixel_colour * w, rcond=-1)[0]
        residuals[p] = pixel_colour * w - np.matmul(bled_codes[:, gene] * w[:, np.newaxis], coefs[p])
    residuals = residuals / weight

    return residuals.astype(np.float32), coefs.astype(np.float32)


def get_best_gene_base(residual_pixel_color: npt.NDArray, all_bled_codes: npt.NDArray,
                       norm_shift: float, score_thresh: float, inverse_var: npt.NDArray,
                       ignore_genes: npt.NDArray) -> Tuple[int, bool]:
    """
    Computes the `dot_product_score` between `residual_pixel_color` and each code in `all_bled_codes`. If `best_score` 
    is less than `score_thresh` or if the corresponding `best_gene` is in `ignore_genes`, then `pass_score_thresh` will 
    be False.

    Args:
        residual_pixel_color (`[(n_rounds * n_channels)] ndarray[float]`): residual pixel color from previous iteration 
            of omp.
        all_bled_codes (`[n_genes x (n_rounds * n_channels)] ndarray[float]`): `bled_codes` such that `spot_color` of a 
            gene `g` in round `r` is expected to be a constant multiple of `bled_codes[g, r]`. Includes codes of genes 
            and background.
        norm_shift (float): shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh (float): `dot_product_score` of the best gene for a pixel must exceed this for that gene to be 
            added in the current iteration.
        inverse_var (`[(n_rounds * n_channels)] ndarray[float]`): inverse of variance in each round/channel based on 
            genes fit on previous iteration. Used as `weight_squared` when computing `dot_product_score`.
        ignore_genes (`[n_genes_ignore] ndarray[int]`): if `best_gene` is one of these, `pass_score_thresh` will be 
            `False`.

    Returns:
        - best_gene - The best gene to add next.
        - pass_score_thresh - `True` if `best_score > score_thresh` and `best_gene` not in `ignore_genes`.
    """
    assert residual_pixel_color.ndim == 1, '`residual_pixel_colors` must be one dimensional'
    assert all_bled_codes.ndim == 2, '`all_bled_codes` must be two dimensional'
    assert inverse_var.ndim == 1, '`inverse_var` must be one dimensional'
    
    # Calculate score including background genes as if best gene is background, then stop iteration.
    all_scores = dot_product.dot_product_score(residual_pixel_color[None], all_bled_codes, inverse_var[None], 
                                               norm_shift)[3][0]
    best_gene = np.argmax(np.abs(all_scores))
    # if best_gene is background, set score below score_thresh.
    best_score = all_scores[best_gene] * np.isin(best_gene, ignore_genes, invert=True)
    pass_score_thresh = np.abs(best_score) > score_thresh
    return best_gene, pass_score_thresh


def get_best_gene_first_iter(residual_pixel_colors: np.ndarray, all_bled_codes: np.ndarray,
                             background_coefs: np.ndarray, norm_shift: float, score_thresh: float, alpha: float, 
                             beta: float, background_genes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the `best_gene` to add next based on the dot product score with each `bled_code`.
    If `best_gene` is in `background_genes` or `best_score < score_thresh` then `pass_score_thresh = False`.
    Different for first iteration as no actual non-zero gene coefficients to consider when computing variance
    or genes that can be added which will cause `pass_score_thresh` to be `False`.

    Args:
        residual_pixel_colors (`[n_pixels x (n_rounds * n_channels)] ndarray[float]`): residual pixel color from 
            previous iteration of omp.
        all_bled_codes (`[n_genes x (n_rounds * n_channels)] ndarray[float]`): `bled_codes` such that `spot_color` of a 
            gene `g` in round `r` is expected to be a constant multiple of `bled_codes[g, r]`. Includes codes of genes 
            and background.
        background_coefs (`[n_pixels x n_channels] ndarray[float]`): `coefs[g]` is the weighting for gene 
            `background_genes[g]` found by the omp algorithm. All are non-zero.
        norm_shift (float): shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh (float): `dot_product_score` of the best gene for a pixel must exceed this for that gene to be 
            added in the current iteration.
        alpha (float): Used for `fitting_variance`, by how much to increase variance as genes added.
        beta (float): Used for `fitting_variance`, the variance with no genes added (`coef=0`) is `beta**2`.
        background_genes (`[n_channels] ndarray[int]`): Indices of codes in `all_bled_codes` which correspond to 
            background. If the best gene for pixel `s` is set to one of `background_genes`, `pass_score_thresh[s]` 
            will be `False`.

    Returns:
        - best_gene (`[n_pixels] ndarray[int]`).
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh (`[n_pixels] ndarray[bool]`).
            `True` if `best_score > score_thresh`.
        - background_var (`[n_pixels x (n_rounds * n_channels)] ndarray[float]`).
            Variance in each round/channel based on just the background.
    """
    n_pixels = residual_pixel_colors.shape[0]
    best_genes = np.zeros(n_pixels, dtype=int)
    pass_score_threshes = np.zeros(n_pixels, dtype=bool)
    background_vars = np.square(background_coefs) @ np.square(all_bled_codes[background_genes]) * alpha + beta ** 2
    #TODO: Eliminate for loop with numpy magic
    for p in range(n_pixels):
        best_gene, pass_score_thresh = get_best_gene_base(residual_pixel_colors[p], all_bled_codes, norm_shift, 
                                                          score_thresh, 1 / background_vars[p], background_genes)
        best_genes[p] = best_gene
        pass_score_threshes[p] = pass_score_thresh
    
    return best_genes, pass_score_threshes, background_vars.astype(np.float32)


def get_best_gene(residual_pixel_colors: npt.NDArray, all_bled_codes: npt.NDArray, coefs: npt.NDArray,
                  genes_added: np.array, norm_shift: float, score_thresh: float, alpha: float,
                  background_genes: npt.NDArray,
                  background_var: np.array) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Finds the `best_gene` to add next to each pixel based on the dot product score with each `bled_code`.
    If `best_gene[s]` is in `background_genes`, already in `genes_added[s]` or `best_score[s] < score_thresh`,
    then `pass_score_thresh[s] = False`.

    Args:
        residual_pixel_colors: `float [n_pixels x (n_rounds * n_channels)]`.
            Residual pixel colors from previous iteration of omp.
        all_bled_codes: `float [n_genes x (n_rounds * n_channels)]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
            Includes codes of genes and background.
        coefs: `float [n_pixels x n_genes_added]`.
            `coefs[s, g]` is the weighting of pixel `s` for gene `genes_added[g]` found by the omp algorithm on previous
            iteration. All are non-zero.
        genes_added: `int [n_pixels x n_genes_added]`
            Indices of genes added to each pixel from previous iteration of omp.
            If the best gene for pixel `s` is set to one of `genes_added[s]`, `pass_score_thresh[s]` will be False.
        norm_shift: shift to apply to normalisation of spot_colors to limit boost of weak spots.
        score_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added in the current iteration.
        alpha: Used for `fitting_variance`, by how much to increase variance as genes added.
        background_genes: `int [n_channels]`.
            Indices of codes in all_bled_codes which correspond to background.
            If the best gene for pixel `s` is set to one of `background_genes`, `pass_score_thresh[s]` will be False.
        background_var: `float [n_pixels x (n_rounds * n_channels)]`.
            Contribution of background genes to variance (which does not change throughout omp iterations)  i.e.
            `background_coefs**2 @ all_bled_codes[background_genes]**2 * alpha + beta ** 2`.

    Returns:
        - best_gene - `int [n_pixels]`.
            `best_gene[s]` is the best gene to add to pixel `s` next.
        - pass_score_thresh - `bool [n_pixels]`.
            `True` if `best_score > score_thresh`.
        - inverse_var - `float [n_pixels x (n_rounds * n_channels)]`.
            Inverse of variance of each pixel in each round/channel based on genes fit on previous iteration.
            Includes both background and gene contribution.

    Notes:
        - The variance computed is based on maximum likelihood estimation - it accounts for all genes and background 
            fit in each round/channel. The more genes added, the greater the variance so if the inverse is used as a 
            weighting for omp fitting or choosing the next gene, the rounds/channels which already have genes in will 
            contribute less.
    """
    n_pixels, n_rounds_channels = residual_pixel_colors.shape
    best_genes = np.zeros((n_pixels), dtype=int)
    pass_score_threshes = np.zeros((n_pixels), dtype=bool)
    inverse_vars = np.zeros((n_pixels, n_rounds_channels), dtype=np.float32)

    #TODO: Eliminate for loop with numpy magic
    for p in range(n_pixels):
        inverse_var = 1 / (np.square(coefs[p]) @ np.square(all_bled_codes[genes_added[p]]) * alpha + background_var[p])
        # calculate score including background genes as if best gene is background, then stop iteration.
        best_gene, pass_score_thresh = get_best_gene_base(residual_pixel_colors[p], all_bled_codes, norm_shift, 
                                                          score_thresh, inverse_var, 
                                                          np.append(background_genes, genes_added[p]))
        best_genes[p] = best_gene
        pass_score_threshes[p] = pass_score_thresh
        inverse_vars[p] = inverse_var

    return best_genes, pass_score_threshes, inverse_vars


def get_all_coefs(pixel_colors: npt.NDArray, bled_codes: npt.NDArray, background_shift: float, dp_shift: float, 
                  dp_thresh: float, alpha: float, beta: float, max_genes: int, weight_coef_fit: bool = False) \
                    -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    This performs omp on every pixel, the stopping criterion is that the dot_product_score when selecting the next gene 
    to add exceeds dp_thresh or the number of genes added to the pixel exceeds max_genes.

    Args:
        pixel_colors: `float [n_pixels x n_rounds x n_channels]`.
            Pixel colors normalised to equalise intensities between channels (and rounds).
        bled_codes: `float [n_genes x n_rounds x n_channels]`.
            `bled_codes` such that `spot_color` of a gene `g`
            in round `r` is expected to be a constant multiple of `bled_codes[g, r]`.
        background_shift: When fitting background,
            this is applied to weighting of each background vector to limit boost of weak pixels.
        dp_shift: When finding `dot_product_score` between residual `pixel_colors` and `bled_codes`,
            this is applied to normalisation of `pixel_colors` to limit boost of weak pixels.
        dp_thresh: `dot_product_score` of the best gene for a pixel must exceed this
            for that gene to be added at each iteration.
        alpha: Used for `fitting_standard_deviation`, by how much to increase variance as genes added.
        beta: Used for `fitting_standard_deviation`, the variance with no genes added (`coef=0`) is `beta**2`.
        max_genes: Maximum number of genes that can be added to a pixel i.e. number of iterations of OMP.
        weight_coef_fit: If False, coefs are found through normal least squares fitting.
            If True, coefs are found through weighted least squares fitting using 1/sigma as the weight factor.

    Returns:
        - gene_coefs - `float32 [n_pixels x n_genes]`.
            `gene_coefs[s, g]` is the weighting of pixel `s` for gene `g` found by the omp algorithm. Most are zero.
        - background_coefs - `float32 [n_pixels x n_channels]`.
            coefficient value for each background vector found for each pixel.

    Notes:
        - Background vectors are fitted first and then not updated again.
    """
    n_pixels = pixel_colors.shape[0]

    rng = np.random.RandomState(0)
    check_spot = rng.randint(n_pixels)
    diff_to_int = np.round(pixel_colors[check_spot]).astype(int) - pixel_colors[check_spot]
    if np.abs(diff_to_int).max() == 0:
        raise ValueError(f"pixel_coefs should be found using normalised pixel_colors."
                         f"\nBut for pixel {check_spot}, pixel_colors given are integers indicating they are "
                         f"the raw intensities.")

    n_genes, n_rounds, n_channels = bled_codes.shape
    if not utils.errors.check_shape(pixel_colors, [n_pixels, n_rounds, n_channels]):
        raise utils.errors.ShapeError('pixel_colors', pixel_colors.shape, (n_pixels, n_rounds, n_channels))
    no_verbose = n_pixels < 1000  # show progress bar with more than 1000 pixels.

    # Fit background and override initial pixel_colors
    gene_coefs = np.zeros((n_pixels, n_genes), dtype=np.float32)  # coefs of all genes and background
    pixel_colors, background_coefs, background_codes = call_spots.fit_background(pixel_colors, background_shift)

    background_genes = np.arange(n_genes, n_genes + n_channels)

    # colors and codes for get_best_gene function
    # Includes background as if background is the best gene, iteration ends.
    # uses residual color as used to find next gene to add.
    bled_codes = bled_codes.reshape((n_genes, -1))
    all_codes = np.concatenate((bled_codes, background_codes.reshape(n_channels, -1)))
    bled_codes = bled_codes.transpose()

    # colors and codes for fit_coefs function (No background as this is not updated again).
    # always uses post background color as coefficients for all genes re-estimated at each iteration.
    pixel_colors = pixel_colors.reshape((n_pixels, -1))

    continue_pixels = np.arange(n_pixels)
    with tqdm(total=max_genes, disable=no_verbose) as pbar:
        pbar.set_description('Finding OMP coefficients for each pixel')
        for i in range(max_genes):
            if i == 0:
                # Background coefs don't change, hence contribution to variance won't either.
                added_genes, pass_score_thresh, background_variance = \
                    get_best_gene_first_iter(pixel_colors, all_codes, background_coefs, dp_shift,
                                             dp_thresh, alpha, beta, background_genes)
                inverse_var = 1 / background_variance
                pixel_colors = pixel_colors.transpose()
            else:
                # only continue with pixels for which dot product score exceeds threshold
                i_added_genes, pass_score_thresh, inverse_var = \
                    get_best_gene(residual_pixel_colors, all_codes, i_coefs, added_genes, dp_shift,
                                  dp_thresh, alpha, background_genes, background_variance)

                # For pixels with at least one non-zero coef, add to final gene_coefs when fail the thresholding.
                fail_score_thresh = np.invert(pass_score_thresh)
                # gene_coefs[np.asarray(continue_pixels[fail_score_thresh])] = np.asarray(i_coefs[fail_score_thresh])
                gene_coefs[np.asarray(continue_pixels[fail_score_thresh])[:, np.newaxis],
                           np.asarray(added_genes[fail_score_thresh])] = np.asarray(i_coefs[fail_score_thresh])

            continue_pixels = continue_pixels[pass_score_thresh]
            n_continue = np.size(continue_pixels)
            pbar.set_postfix({'n_pixels': n_continue})
            if n_continue == 0:
                break
            if i == 0:
                added_genes = added_genes[pass_score_thresh, np.newaxis]
            else:
                added_genes = np.hstack((added_genes[pass_score_thresh], i_added_genes[pass_score_thresh, np.newaxis]))
            pixel_colors = pixel_colors[:, pass_score_thresh]
            background_variance = background_variance[pass_score_thresh]
            inverse_var = inverse_var[pass_score_thresh]

            # Maybe add different fit_coefs for i==0 i.e. can do multiple pixels at once for same gene added.
            if weight_coef_fit:
                residual_pixel_colors, i_coefs = fit_coefs_weight(bled_codes, pixel_colors, added_genes,
                                                                  np.sqrt(inverse_var))
            else:
                residual_pixel_colors, i_coefs = fit_coefs(bled_codes, pixel_colors, added_genes)

            if i == max_genes-1:
                # Add pixels to final gene_coefs when reach end of iteration.
                gene_coefs[np.asarray(continue_pixels)[:, np.newaxis], np.asarray(added_genes)] = np.asarray(i_coefs)

            pbar.update(1)
    pbar.close()

    return gene_coefs.astype(np.float32), np.asarray(background_coefs, dtype=np.float32)


def get_pixel_colours(nbp_basic: NotebookPage, nbp_file: NotebookPage, nbp_extract: NotebookPage, tile: int, 
                      z_chunk: int, z_chunk_size: int, transform: np.ndarray, colour_norm_factor: np.ndarray, 
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the normalised pixel colours and their pixel positions for one z chunk.

    Args:
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_extract (NotebookPage): 'extract' notebook page.
        tile (int): tile index.
        z_chunk (int): z chunk index.
        z_chunk_size (int): z chunk size
        transform (`[n_tiles x n_rounds x n_channels x 4 x 3] ndarray[float]`): `transform[t, r, c]` is the affine 
            transform to get from tile `t`, `ref_round`, `ref_channel` to tile `t`, round `r`, channel `c`.
        colour_norm_factor (`[n_rounds x n_channels] ndarray[float]`): Normalisation factors to divide colours by to 
            equalise channel intensities.

    Returns:
        - (`[n_pixels x 3] ndarray[int16]`): `pixel_yxz_tz` is the y, x and z pixel positions of the pixel colours 
            found. 
        - (`[n_pixels x n_rounds x n_channels] ndarray[float32]`): `pixel_colours_tz` contains the colours for each 
            pixel.
    """
    def get_z_plane_colours(z_index: int, q: multiprocessing.Queue) -> None:
        no_output = None, None
        if nbp_basic.use_preseq:
            pixel_colors_t1, pixel_yxz_t1, _ = \
                spot_colors.base.get_spot_colors(
                    spot_colors.base.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, 
                                                   np.arange(z_index, z_index + 1)), 
                    int(tile), transform, nbp_file, nbp_basic, nbp_extract, return_in_bounds=True)
        else:
            pixel_colors_t1, pixel_yxz_t1 = \
                spot_colors.base.get_spot_colors(
                    spot_colors.base.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, 
                                                   np.arange(z_index, z_index + 1)), 
                    int(tile), transform, nbp_file, nbp_basic, nbp_extract, return_in_bounds=True)
        
        pixel_colors_t1 = pixel_colors_t1.astype(np.float32) / colour_norm_factor
        if pixel_colors_t1.shape[0] == 0:
            q.put(list(no_output))
            return
        q.put([pixel_yxz_t1.astype(np.int16), pixel_colors_t1.astype(np.float32)])
        return
    
    z_min, z_max = z_chunk * z_chunk_size, min((z_chunk + 1) * z_chunk_size, len(nbp_basic.use_z))
    pixel_yxz_tz = np.zeros((0, 3), dtype=np.int16)
    pixel_colors_tz = np.zeros((0, len(nbp_basic.use_rounds), len(nbp_basic.use_channels)), dtype=np.float32)
    
    queue = multiprocessing.Queue()
    
    for z_plane in range(z_min, z_max):
        new_process = multiprocessing.Process(target=get_z_plane_colours, args=(z_plane, queue, ))
        new_process.start()
    
    for _ in range(z_min, z_max):
        pixel_yxz_t1, pixel_colors_t1 = queue.get()
        
        if pixel_yxz_t1 is None or pixel_colors_t1 is None:
            continue
        pixel_yxz_tz = np.append(pixel_yxz_tz, pixel_yxz_t1, axis=0)
        pixel_colors_tz = np.append(pixel_colors_tz, pixel_colors_t1, axis=0)
    
    queue.close()
    
    return pixel_yxz_tz, pixel_colors_tz


def get_pixel_coefs_yxz(nbp_basic: NotebookPage, nbp_file: NotebookPage, nbp_extract: NotebookPage, config: dict, 
                        tile: int, use_z: List[int], z_chunk_size: int, n_genes: int, 
                        transform: Union[np.ndarray, np.ndarray], color_norm_factor: Union[np.ndarray, np.ndarray], 
                        initial_intensity_thresh: float, bled_codes: Union[np.ndarray, np.ndarray], 
                        dp_norm_shift: Union[int, float], n_threads: Optional[int] = None) -> Tuple[np.ndarray, Any]:
    """
    Get each pixel OMP coefficients for a particular tile.
    
    Args:
        nbp_basic (NotebookPage): notebook page for 'basic_info'.
        nbp_file (NotebookPage): notebook page for 'file_names'.
        nbp_extract (NotebookPage): notebook page for 'extract'.
        config (dict): config settings for 'omp'.
        tile (int): tile index.
        use_z (list of int): list of z planes to calculate on.
        z_chunk_size (int): size of each z chunk.
        n_genes (int): the number of genes.
        transform (`[n_tiles x n_rounds x n_channels x 4 x 3] ndarray[float]`): `transform[t, r, c]` is the affine 
            transform to get from tile `t`, `ref_round`, `ref_channel` to tile `t`, round `r`, channel `c`.
        color_norm_factor (`[n_rounds x n_channels] ndarray[float]`): Normalisation factors to divide colours by to 
            equalise channel intensities.
        initial_intensity_thresh (float): pixel intensity threshold, only keep ones above the threshold to save memory 
            and storage space.
        bled_codes (`[n_genes x n_rounds x n_channels] ndarray[float]`): bled codes.
        dp_norm_shift (int or float): when finding `dot_product_score` between residual `pixel_colors` and 
            `bled_codes`, this is applied to normalisation of `pixel_colors` to limit boost of weak pixels.
        n_threads (int, optional): number of threads to use when computing each z chunk. Default: as many as possible.

    Returns:
        - (`[n_pixels x 3] ndarray[int]`): `pixel_yxz_t` is the y, x and z pixel positions of the gene coefficients 
            found.
        - (`[n_pixels x n_genes]`): `pixel_coefs_t` contains the gene coefficients for each pixel.
    """
    def compute_z_chunk(z_chunk: int, q: multiprocessing.Queue) -> None:
        no_output = None, None
        z_min, z_max = z_chunk * z_chunk_size, min((z_chunk + 1) * z_chunk_size, len(use_z))
        if nbp_basic.use_preseq:
            pixel_colors_tz, pixel_yxz_tz, bg_colours = \
                spot_colors.get_spot_colors(
                    spot_colors.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, np.arange(z_min, z_max)), 
                    int(tile), transform, nbp_file, nbp_basic, nbp_extract, return_in_bounds=True)
        else:
            pixel_colors_tz, pixel_yxz_tz = \
                spot_colors.get_spot_colors(
                    spot_colors.all_pixel_yxz(nbp_basic.tile_sz, nbp_basic.tile_sz, np.arange(z_min, z_max)), 
                    int(tile), transform, nbp_file, nbp_basic, nbp_extract, return_in_bounds=True)
        if pixel_colors_tz.shape[0] == 0:
            q.put(list(no_output))
            return
        pixel_colors_tz = pixel_colors_tz / color_norm_factor
        np.asarray(pixel_colors_tz, dtype=np.float32)

        # Only keep pixels with significant absolute intensity to save memory.
        # absolute because important to find negative coefficients as well.
        pixel_intensity_tz = call_spots.get_spot_intensity(np.abs(pixel_colors_tz))
        keep = pixel_intensity_tz > initial_intensity_thresh
        if not keep.any():
            q.put(list(no_output))
            return
        pixel_colors_tz = pixel_colors_tz[keep]
        pixel_yxz_tz = pixel_yxz_tz[keep]
        del pixel_intensity_tz, keep

        pixel_coefs_tz = get_all_coefs(pixel_colors_tz, bled_codes, 0, dp_norm_shift, config['dp_thresh'], config['alpha'], 
                                    config['beta'], config['max_genes'], config['weight_coef_fit'])[0]
        pixel_coefs_tz = np.asarray(pixel_coefs_tz)
        del pixel_colors_tz
        # Only keep pixels for which at least one gene has non-zero coefficient.
        keep = (np.abs(pixel_coefs_tz).max(axis=1) > 0).nonzero()[0]  # nonzero as is sparse matrix.
        if len(keep) == 0:
            q.put(list(no_output))
            return
        
        q.put([pixel_yxz_tz[keep], pixel_coefs_tz[keep]])
        return


    pixel_yxz_t = np.zeros((0, 3), dtype=np.int16)
    pixel_coefs_t = scipy.sparse.csr_matrix(np.zeros((0, n_genes), dtype=np.float32))

    n_threads = utils.threads.get_available_threads()

    z_chunks = len(use_z) // z_chunk_size + 1
    processes = []
    queue = multiprocessing.Queue()
    for z_chunk in range(z_chunks):
        print(f"z_chunk {z_chunk + 1}/{z_chunks}")
        # While iterating through tiles, only save info for rounds/channels using
        # - add all rounds/channels back in later. This returns colors in use_rounds/channels only and no invalid.
        processes.append(multiprocessing.Process(target=compute_z_chunk, args=(z_chunk, queue)))
        
        if len(processes) == n_threads or z_chunk + 1 == z_chunks:
            # Start all subprocesses and append the outputs
            [p.start() for p in processes]
            
            for p in processes:
                pixel_yxz_tz, pixel_coefs_tz = queue.get()
                
                if pixel_yxz_tz is not None and pixel_coefs_tz is not None:        
                    pixel_yxz_t = np.append(pixel_yxz_t, np.asarray(pixel_yxz_tz), axis=0)
                    pixel_coefs_t = scipy.sparse.vstack((pixel_coefs_t, scipy.sparse.csr_matrix(pixel_coefs_tz)))
                del pixel_yxz_tz, pixel_coefs_tz
            processes = []

    return pixel_yxz_t, pixel_coefs_t
