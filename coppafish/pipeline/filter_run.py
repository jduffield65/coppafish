import os
import time
import warnings
import numpy as np
from tqdm import tqdm
import numpy.typing as npt
from typing import Optional, Tuple

from .. import utils, extract
from ..register import preprocessing
from ..utils import tiles_io
from ..filter import deconvolution
from ..setup.notebook import NotebookPage, Notebook


def run_filter(
    config: dict,
    nbp_file: NotebookPage,
    nbp_basic: NotebookPage,
    nbp_scale: NotebookPage,
    nbp_extract: NotebookPage,
    image_t_raw: Optional[npt.NDArray[np.uint16]] = None,
    return_filtered_image: Optional[bool] = False, 
) -> Tuple[NotebookPage, NotebookPage, Optional[npt.NDArray[np.uint16]]]:
    """
    Read in extracted raw images, filter them, then re-save in a different location.

    Args:
        config (dict): dictionary obtained from 'filter' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.
        nbp_scale (NotebookPage): 'scale' notebook page.
        nbp_extract (NotebookPage): 'extract' notebook page.
        image_t_raw (`(n_rounds x n_channels x nz x ny x nx) ndarray[uint16]`, optional): extracted image for single
            tile. Can only be used for a single tile notebooks. Default: not given.

    Returns:
        - NotebookPage: 'filter' notebook page.
        - NotebookPage: 'filter_debug' notebook page.
        - `(n_rounds x n_channels x nz x ny x nx) ndarray[uint16]` or None: if `nbp_basic.use_tiles` is a single tile,
            returns all saved tile images. Otherwise, returns None.

    Notes:
        - See `'filter'` and `'filter_debug'` sections of `notebook_comments.json` file for description of variables.
    """
    # TODO: Refactor this function to make it more readable
    if not nbp_basic.is_3d:
        NotImplementedError(f"2d coppafish is not stable, very sorry! :9")
    if image_t_raw is not None:
        assert len(nbp_basic.use_tiles) == 1, "If image_t is given, the notebook must contain a single tile"
    if return_filtered_image:
        assert len(nbp_basic.use_tiles) == 1, "To return a filtered image, filter must be run on a single tile"

    nbp = NotebookPage("filter")
    nbp_debug = NotebookPage("filter_debug")
    nbp.software_version = utils.system.get_software_verison()
    nbp.revision_hash = utils.system.get_git_revision_hash()

    start_time = time.time()
    file_type = nbp_extract.file_type
    # get rounds to iterate over
    use_channels_anchor = [c for c in [nbp_basic.dapi_channel, nbp_basic.anchor_channel] if c is not None]
    use_channels_anchor.sort()
    if nbp_basic.use_anchor:
        # always have anchor as first round after imaging rounds
        round_files = nbp_file.round + [nbp_file.anchor]
        use_rounds = np.arange(len(round_files))
        n_images = (
            (len(use_rounds) - 1) * len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)
            + len(nbp_basic.use_tiles) * len(use_channels_anchor)
            + len(nbp_basic.use_tiles) * len(nbp_basic.use_rounds) * nbp_extract.continuous_dapi
        )
    else:
        round_files = nbp_file.round
        use_rounds = nbp_basic.use_rounds
        n_images = len(use_rounds) * len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)
    if return_filtered_image:
        image_t = np.zeros(
            (
                nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
                nbp_basic.n_channels,
                nbp_basic.nz,
                nbp_basic.tile_sz,
                nbp_basic.tile_sz,
            ),
            dtype=np.uint16,
        )
    else:
        image_t = None

    # initialise output of this part of pipeline as 'vars' key
    nbp.auto_thresh = np.zeros(
        (
            nbp_basic.n_tiles,
            nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
            nbp_basic.n_channels,
        ),
        dtype=int,
    )
    nbp.hist_values = np.arange(
        -nbp_basic.tile_pixel_value_shift,
        np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift + 2,
        1,
    )
    nbp.hist_counts = np.zeros(
        (
            len(nbp.hist_values),
            nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
            nbp_basic.n_channels,
        ),
        dtype=int,
    )

    # initialise debugging info as 'debug' page
    nbp_debug.n_clip_pixels = np.zeros_like(nbp.auto_thresh, dtype=int)
    nbp_debug.clip_extract_scale = np.zeros_like(nbp.auto_thresh)
    nbp_debug.pixel_unique_values = np.full(
        (
            nbp_basic.n_tiles,
            nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
            nbp_basic.n_channels,
            np.iinfo(np.uint16).max,
        ),
        fill_value=0,
        dtype=int, 
    )
    nbp_debug.pixel_unique_counts = nbp_debug.pixel_unique_values.copy()

    # If we have a pre-sequencing round, add this to round_files at the end
    if nbp_basic.use_preseq:
        round_files = round_files + [nbp_file.pre_seq]
        use_rounds = np.arange(len(round_files))
        pre_seq_round = len(round_files) - 1
        n_images += len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)
    else:
        pre_seq_round = None

    n_clip_error_images = 0
    if config["n_clip_error"] is None:
        # default is 1% of pixels on single z-plane
        config["n_clip_error"] = int(nbp_basic.tile_sz * nbp_basic.tile_sz / 100)

    if nbp_basic.is_3d:
        nbp_debug.z_info = int(np.floor(nbp_basic.nz / 2))  # central z-plane to get info from.
    else:
        nbp_debug.z_info = 0
    hist_bin_edges = np.concatenate((nbp.hist_values - 0.5, nbp.hist_values[-1:] + 0.5))
    nbp_debug.r_dapi = config["r_dapi"]
    filter_kernel = utils.morphology.hanning_diff(nbp_scale.r1, nbp_scale.r2)
    if nbp_debug.r_dapi is not None:
        filter_kernel_dapi = utils.strel.disk(nbp_debug.r_dapi)
    else:
        filter_kernel_dapi = None

    if nbp_scale.r_smooth is not None:
        # smooth_kernel = utils.strel.fspecial(*tuple(nbp_scale.r_smooth]))
        smooth_kernel = np.ones(tuple(np.array(nbp_scale.r_smooth, dtype=int) * 2 - 1))
        smooth_kernel = smooth_kernel / np.sum(smooth_kernel)
    if config["deconvolve"]:
        if not os.path.isfile(nbp_file.psf):
            (
                spot_images,
                config["psf_intensity_thresh"],
                psf_tiles_used,
            ) = deconvolution.get_psf_spots(
                nbp_file,
                nbp_basic,
                nbp_extract, 
                nbp_basic.anchor_round,
                nbp_basic.use_tiles,
                nbp_basic.anchor_channel,
                nbp_basic.use_z,
                config["psf_detect_radius_xy"],
                config["psf_detect_radius_z"],
                config["psf_min_spots"],
                config["psf_intensity_thresh"],
                config["auto_thresh_multiplier"],
                config["psf_isolation_dist"],
                config["psf_shape"],
            )
            psf = deconvolution.get_psf(spot_images, config["psf_annulus_width"])
            np.save(nbp_file.psf, np.moveaxis(psf, 2, 0))  # save with z as first axis
        else:
            # Know psf only computed for 3D pipeline hence know ndim=3
            psf = np.moveaxis(np.load(nbp_file.psf), 0, 2)  # Put z to last index
            psf_tiles_used = None
        # normalise psf so min is 0 and max is 1.
        psf = psf - psf.min()
        psf = psf / psf.max()
        pad_im_shape = (
            np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z)])
            + np.array(config["wiener_pad_shape"]) * 2
        )
        wiener_filter = deconvolution.get_wiener_filter(psf, pad_im_shape, config["wiener_constant"])
        nbp_debug.psf = psf
        if config["psf_intensity_thresh"] is not None:
            config["psf_intensity_thresh"] = int(config["psf_intensity_thresh"])
        nbp_debug.psf_intensity_thresh = config["psf_intensity_thresh"]
        nbp_debug.psf_tiles_used = psf_tiles_used
    else:
        nbp_debug.psf = None
        nbp_debug.psf_intensity_thresh = None
        nbp_debug.psf_tiles_used = None

    with tqdm(
        total=n_images,
        desc=f"Filtering extracted {nbp_extract.file_type} tiles",
    ) as pbar:
        for r in use_rounds:
            if r == nbp_basic.anchor_round:
                n_clip_error_images = 0  # reset for anchor as different scale used.
                scale = nbp_scale.scale_anchor
                use_channels = use_channels_anchor
            else:
                scale = nbp_scale.scale
                use_channels = nbp_basic.use_channels + [nbp_basic.dapi_channel] * nbp_extract.continuous_dapi

            for t in nbp_basic.use_tiles:
                if not nbp_basic.is_3d:
                    # for 2d all channels in same file
                    file_exists_raw = tiles_io.image_exists(nbp_file.tile_unfiltered[t][r], file_type)
                    if file_exists_raw:
                        # mmap load in image for all channels if tiff exists
                        im_all_channels_2d = np.load(nbp_file.tile_unfiltered[t][r], mmap_mode="r")
                    else:
                        # Only save 2d data when all channels collected
                        # For channels not used, keep all pixels 0.
                        im_all_channels_2d = np.zeros(
                            (
                                nbp_basic.n_channels,
                                nbp_basic.tile_sz,
                                nbp_basic.tile_sz,
                            ),
                            dtype=np.int32,
                        )
                for c in use_channels:
                    if c == nbp_basic.dapi_channel:
                        max_tiff_pixel_value = np.iinfo(np.uint16).max
                    else:
                        max_tiff_pixel_value = np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift
                    if nbp_basic.is_3d:
                        if r != pre_seq_round:
                            file_path = nbp_file.tile[t][r][c]
                            file_exists = tiles_io.image_exists(file_path, file_type)
                            file_path_raw = nbp_file.tile_unfiltered[t][r][c]
                            file_exists_raw = tiles_io.image_exists(file_path_raw, file_type)
                        else:
                            file_path = nbp_file.tile[t][r][c]
                            file_path = file_path[: file_path.index(file_type)] + "_raw" + file_type
                            file_exists = tiles_io.image_exists(file_path, file_type)
                            file_path_raw = nbp_file.tile_unfiltered[t][r][c]
                            file_path_raw = file_path_raw[: file_path_raw.index(file_type)] + "_raw" + file_type
                            file_exists_raw = tiles_io.image_exists(file_path_raw, file_type)
                        assert (
                            file_exists_raw
                        ), f"Raw, extracted file at\n\t{file_path_raw}\nwas not found, were the extracted files moved?"

                    pbar.set_postfix(
                        {
                            "round": r,
                            "tile": t,
                            "channel": c,
                            "exists": str(file_exists),
                        }
                    )

                    if file_exists:
                        if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
                            pass
                        else:
                            # Only need to load in mid-z plane if 3D.
                            if nbp_basic.is_3d:
                                im = tiles_io.load_image(
                                    nbp_file,
                                    nbp_basic,
                                    file_type,
                                    t,
                                    r,
                                    c,
                                    yxz=[None, None, nbp_debug.z_info],
                                    suffix="_raw" if r == pre_seq_round else "",
                                    apply_shift=False, 
                                )
                                if not (r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel):
                                    im = preprocessing.apply_image_shift(im, -nbp_basic.tile_pixel_value_shift)
                            else:
                                im = im_all_channels_2d[c].astype(np.int32) - nbp_basic.tile_pixel_value_shift
                            (
                                nbp.auto_thresh[t, r, c],
                                hist_counts_trc,
                                nbp_debug.n_clip_pixels[t, r, c],
                                nbp_debug.clip_extract_scale[t, r, c],
                            ) = extract.base.get_extract_info(
                                im,
                                config["auto_thresh_multiplier"],
                                hist_bin_edges,
                                max_tiff_pixel_value,
                                scale,
                            )
                            if r != nbp_basic.anchor_round:
                                # Does hist counts ad all tiles?
                                nbp.hist_counts[:, r, c] += hist_counts_trc
                    if not file_exists:
                        if image_t_raw is None:
                            im_raw = tiles_io._load_image(file_path_raw, file_type)
                        else:
                            im_raw = image_t_raw[r, c]
                        # zyx -> yxz
                        im_raw = im_raw.transpose((1, 2, 0))

                        if not nbp_basic.is_3d:
                            im_raw = extract.focus_stack(im_raw)
                        im, bad_columns = extract.strip_hack(im_raw)  # find faulty columns
                        del im_raw
                        # This will deconcolve dapis as well, but I think that is what we want.
                        if config["deconvolve"]:
                            im = deconvolution.wiener_deconvolve(im, config["wiener_pad_shape"], wiener_filter)
                        if c == nbp_basic.dapi_channel:
                            if filter_kernel_dapi is not None:
                                im = utils.morphology.top_hat(im, filter_kernel_dapi)
                        elif c != nbp_basic.dapi_channel:
                            # im converted to float in convolve_2d so no point changing dtype beforehand.
                            im = utils.morphology.convolve_2d(im, filter_kernel) * scale
                            if nbp_scale.r_smooth is not None:
                                # oa convolve uses lots of memory and much slower here.
                                im = utils.morphology.imfilter(im, smooth_kernel, oa=False)
                            im[:, bad_columns] = 0
                            # get_info is quicker on int32 so do this conversion first.
                            im = np.rint(im, np.zeros_like(im, dtype=np.int32), casting="unsafe")
                            # only use image unaffected by strip_hack to get information from tile
                            good_columns = np.setdiff1d(np.arange(nbp_basic.tile_sz), bad_columns)
                            (
                                nbp.auto_thresh[t, r, c],
                                hist_counts_trc,
                                nbp_debug.n_clip_pixels[t, r, c],
                                nbp_debug.clip_extract_scale[t, r, c],
                            ) = extract.get_extract_info(
                                im[:, good_columns],
                                config["auto_thresh_multiplier"],
                                hist_bin_edges,
                                max_tiff_pixel_value,
                                scale,
                                nbp_debug.z_info,
                            )

                            # Deal with pixels outside uint16 range when saving
                            if c != nbp_basic.dapi_channel and nbp_debug.n_clip_pixels[t, r, c] > config["n_clip_warn"]:
                                warnings.warn(
                                    f"\nTile {t}, round {r}, channel {c} has "
                                    f"{nbp_debug.n_clip_pixels[t, r, c]} pixels\n"
                                    f"that will be clipped when converting to uint16."
                                )
                            if (
                                c != nbp_basic.dapi_channel
                                and nbp_debug.n_clip_pixels[t, r, c] > config["n_clip_error"]
                            ):
                                n_clip_error_images += 1
                                message = (
                                    f"\nNumber of images for which more than {config['n_clip_error']} pixels "
                                    f"clipped in conversion to uint16 is {n_clip_error_images}."
                                )
                                if n_clip_error_images >= config["n_clip_error_images_thresh"]:
                                    # create new Notebook to save info obtained so far
                                    nb_fail_name = os.path.join(
                                        nbp_file.output_dir,
                                        "notebook_extract_error.npz",
                                    )
                                    nb_fail = Notebook(nb_fail_name, None)
                                    # change names of pages so can add extra properties not in json file.
                                    nbp.name = "extract_fail"
                                    nbp_debug.name = "extract_debug_fail"
                                    # record where failure occurred
                                    nbp.fail_trc = np.array([t, r, c])
                                    nbp_debug.fail_trc = np.array([t, r, c])
                                    nb_fail += nbp
                                    nb_fail += nbp_debug
                                    raise ValueError(f"{message}\nResults up till now saved as {nb_fail_name}.")
                                else:
                                    warnings.warn(
                                        f"{message}\nWhen this reaches {config['n_clip_error_images_thresh']}"
                                        f", the extract step of the algorithm will be interrupted."
                                    )

                            if r != nbp_basic.anchor_round:
                                nbp.hist_counts[:, r, c] += hist_counts_trc
                        # delay gaussian blurring of preseq until after reg to give it a better chance
                        if nbp_basic.is_3d:
                            saved_im = tiles_io.save_image(
                                nbp_file,
                                nbp_basic,
                                file_type,
                                im,
                                t,
                                r,
                                c,
                                suffix="_raw" if r == pre_seq_round else "",
                                num_rotations=config["num_rotations"],
                            )
                            if return_filtered_image:
                                image_t[r, c] = saved_im
                            pixel_unique_values, pixel_unique_counts = np.unique(saved_im, return_counts=True)
                            nbp_debug.pixel_unique_values[t][r][c][: pixel_unique_values.size] = pixel_unique_values
                            nbp_debug.pixel_unique_counts[t][r][c][: pixel_unique_counts.size] = pixel_unique_counts
                            del saved_im
                        else:
                            im_all_channels_2d[c] = im
                    pbar.update(1)
                if not nbp_basic.is_3d:
                    tiles_io.save_image(
                        nbp_file,
                        nbp_basic,
                        file_type,
                        im_all_channels_2d,
                        t,
                        r,
                        suffix="_raw" if r == pre_seq_round else "",
                    )
    # Add a variable for bg_scale (actually computed in register)
    nbp.bg_scale = None
    end_time = time.time()
    nbp_debug.time_taken = end_time - start_time
    return nbp, nbp_debug, image_t
