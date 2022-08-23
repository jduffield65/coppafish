from .. import utils, extract
import numpy as np
import os
from tqdm import tqdm
from ..setup.notebook import NotebookPage, Notebook
from typing import Tuple
import warnings


def extract_and_filter(config: dict, nbp_file: NotebookPage,
                       nbp_basic: NotebookPage) -> Tuple[NotebookPage, NotebookPage]:
    """
    This reads in images from the raw `nd2` files, filters them and then saves them as npy files in the tile directory.
    Also gets `auto_thresh` for use in turning images to point clouds and `hist_values`, `hist_counts` required for
    normalisation between channels.

    Returns the `extract` and `extract_debug` notebook pages.

    See `'extract'` and `'extract_debug'` sections of `notebook_comments.json` file
    for description of the variables in each page.

    Args:
        config: Dictionary obtained from `'extract'` section of config file.
        nbp_file: `file_names` notebook page
        nbp_basic: `basic_info` notebook page

    Returns:
        - `NotebookPage[extract]` - Page containing `auto_thresh` for use in turning images to point clouds and
            `hist_values`, `hist_counts` required for normalisation between channels.
        - `NotebookPage[extract_debug]` - Page containing variables which are not needed later in the pipeline
            but may be useful for debugging purposes.
    """
    # Check scaling won't cause clipping when saving as uint16
    scale_norm_max = np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift
    if config['scale_norm'] >= scale_norm_max:
        raise ValueError(f"\nconfig['extract']['scale_norm'] = {config['scale_norm']} but it must be below "
                         f"{scale_norm_max}")

    # initialise notebook pages
    if not nbp_basic.is_3d:
        config['deconvolve'] = False  # only deconvolve if 3d pipeline
    nbp = NotebookPage("extract")
    nbp_debug = NotebookPage("extract_debug")
    # initialise output of this part of pipeline as 'vars' key
    nbp.auto_thresh = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
                                nbp_basic.n_channels), dtype=int)
    nbp.hist_values = np.arange(-nbp_basic.tile_pixel_value_shift, np.iinfo(np.uint16).max -
                                nbp_basic.tile_pixel_value_shift + 2, 1)
    nbp.hist_counts = np.zeros((len(nbp.hist_values), nbp_basic.n_rounds, nbp_basic.n_channels), dtype=int)
    hist_bin_edges = np.concatenate((nbp.hist_values - 0.5, nbp.hist_values[-1:] + 0.5))
    # initialise debugging info as 'debug' page
    nbp_debug.n_clip_pixels = np.zeros_like(nbp.auto_thresh, dtype=int)
    nbp_debug.clip_extract_scale = np.zeros_like(nbp.auto_thresh, dtype=np.float32)
    if nbp_basic.is_3d:
        nbp_debug.z_info = int(np.floor(nbp_basic.nz / 2))  # central z-plane to get info from.
    else:
        nbp_debug.z_info = 0

    # update config params in notebook. All optional parameters in config are added to debug page
    if config['r1'] is None:
        config['r1'] = extract.get_pixel_length(config['r1_auto_microns'], nbp_basic.pixel_size_xy)
    if config['r2'] is None:
        config['r2'] = config['r1'] * 2
    if config['r_dapi'] is None:
        if config['r_dapi_auto_microns'] is not None:
            config['r_dapi'] = extract.get_pixel_length(config['r_dapi_auto_microns'], nbp_basic.pixel_size_xy)
    nbp_debug.r1 = config['r1']
    nbp_debug.r2 = config['r2']
    nbp_debug.r_dapi = config['r_dapi']

    filter_kernel = utils.morphology.hanning_diff(nbp_debug.r1, nbp_debug.r2)
    if nbp_debug.r_dapi is not None:
        filter_kernel_dapi = utils.strel.disk(nbp_debug.r_dapi)
    else:
        filter_kernel_dapi = None

    if config['r_smooth'] is not None:
        if len(config['r_smooth']) == 2:
            if nbp_basic.is_3d:
                warnings.warn(f"Running 3D pipeline but only 2D smoothing requested with r_smooth"
                              f" = {config['r_smooth']}.")
        elif len(config['r_smooth']) == 3:
            if not nbp_basic.is_3d:
                raise ValueError("Running 2D pipeline but 3D smoothing requested.")
        else:
            raise ValueError(f"r_smooth provided was {config['r_smooth']}.\n"
                             f"But it needs to be a 2 radii for 2D smoothing or 3 radii for 3D smoothing.\n"
                             f"I.e. it is the wrong shape.")
        if config['r_smooth'][0] > config['r2']:
            raise ValueError(f"Smoothing radius, {config['r_smooth'][0]}, is larger than the outer radius of the\n"
                             f"hanning filter, {config['r2']}, making the filtering step redundant.")

        # smooth_kernel = utils.strel.fspecial(*tuple(config['r_smooth']))
        smooth_kernel = np.ones(tuple(np.array(config['r_smooth'], dtype=int) * 2 - 1))
        smooth_kernel = smooth_kernel / np.sum(smooth_kernel)

        if np.max(config['r_smooth']) == 1:
            warnings.warn('Max radius of smooth filter was 1, so not using.')
            config['r_smooth'] = None

    if config['deconvolve']:
        if not os.path.isfile(nbp_file.psf):
            spot_images, config['psf_intensity_thresh'], psf_tiles_used = \
                extract.get_psf_spots(nbp_file, nbp_basic, nbp_basic.ref_round,
                                      nbp_basic.use_tiles, nbp_basic.ref_channel, nbp_basic.use_z,
                                      config['psf_detect_radius_xy'], config['psf_detect_radius_z'],
                                      config['psf_min_spots'], config['psf_intensity_thresh'],
                                      config['auto_thresh_multiplier'], config['psf_isolation_dist'],
                                      config['psf_shape'])
            psf = extract.get_psf(spot_images, config['psf_annulus_width'])
            np.save(nbp_file.psf, np.moveaxis(psf, 2, 0))  # save with z as first axis
        else:
            # Know psf only computed for 3D pipeline hence know ndim=3
            psf = np.moveaxis(np.load(nbp_file.psf), 0, 2)  # Put z to last index
            psf_tiles_used = None
        # normalise psf so min is 0 and max is 1.
        psf = psf - psf.min()
        psf = psf / psf.max()
        pad_im_shape = np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz]) + \
                       np.array(config['wiener_pad_shape']) * 2
        wiener_filter = extract.get_wiener_filter(psf, pad_im_shape, config['wiener_constant'])
        nbp_debug.psf = psf
        nbp_debug.psf_intensity_thresh = config['psf_intensity_thresh']
        nbp_debug.psf_tiles_used = psf_tiles_used
    else:
        nbp_debug.psf = None
        nbp_debug.psf_intensity_thresh = None
        nbp_debug.psf_tiles_used = None

    # check to see if scales have already been computed
    config['scale'], config['scale_anchor'] = extract.get_scale_from_txt(nbp_file.scale, config['scale'],
                                                                         config['scale_anchor'])

    if config['scale'] is None and len(nbp_file.round) > 0:
        # ensure scale_norm value is reasonable so max pixel value in tiff file is significant factor of max of uint16
        scale_norm_min = (np.iinfo('uint16').max - nbp_basic.tile_pixel_value_shift) / 5
        scale_norm_max = np.iinfo('uint16').max - nbp_basic.tile_pixel_value_shift
        if not scale_norm_min <= config['scale_norm'] <= scale_norm_max:
            raise utils.errors.OutOfBoundsError("scale_norm", config['scale_norm'], scale_norm_min, scale_norm_max)
        # If using smoothing, apply this as well before scal
        if config['r_smooth'] is None:
            smooth_kernel_2d = None
        else:
            if smooth_kernel.ndim == 3:
                # take central plane of smooth filter if 3D as scale is found from a single z-plane.
                smooth_kernel_2d = smooth_kernel[:, :, config['r_smooth'][2] - 1]
            else:
                smooth_kernel_2d = smooth_kernel.copy()
            # smoothing is averaging so to average in 2D, need to re normalise filter
            smooth_kernel_2d = smooth_kernel_2d / np.sum(smooth_kernel_2d)
            if np.max(config['r_smooth'][:2]) <= 1:
                smooth_kernel_2d = None  # If dimensions of 2D kernel are [1, 1] is equivalent to no smoothing
        nbp_debug.scale_tile, nbp_debug.scale_channel, nbp_debug.scale_z, config['scale'] = \
            extract.get_scale(nbp_file, nbp_basic, 0, nbp_basic.use_tiles, nbp_basic.use_channels, nbp_basic.use_z,
                              config['scale_norm'], filter_kernel, smooth_kernel_2d)
    else:
        nbp_debug.scale_tile = None
        nbp_debug.scale_channel = None
        nbp_debug.scale_z = None
        smooth_kernel_2d = None
    nbp_debug.scale = config['scale']
    print(f"Scale: {nbp_debug.scale}")

    # save scale values incase need to re-run
    extract.save_scale(nbp_file.scale, nbp_debug.scale, config['scale_anchor'])

    # get rounds to iterate over
    use_channels_anchor = [c for c in [nbp_basic.dapi_channel, nbp_basic.anchor_channel] if c is not None]
    use_channels_anchor.sort()
    if filter_kernel_dapi is None:
        # If not filtering DAPI, skip over the DAPI channel.
        use_channels_anchor = np.setdiff1d(use_channels_anchor, nbp_basic.dapi_channel)
    if nbp_basic.use_anchor:
        # always have anchor as first round after imaging rounds
        round_files = nbp_file.round + [nbp_file.anchor]
        use_rounds = nbp_basic.use_rounds + [nbp_basic.n_rounds]
        n_images = (len(use_rounds) - 1) * len(nbp_basic.use_tiles) * len(nbp_basic.use_channels) + \
                   len(nbp_basic.use_tiles) * len(use_channels_anchor)
    else:
        round_files = nbp_file.round
        use_rounds = nbp_basic.use_rounds
        n_images = len(use_rounds) * len(nbp_basic.use_tiles) * len(nbp_basic.use_channels)

    n_clip_error_images = 0
    if config['n_clip_error'] is None:
        # default is 1% of pixels on single z-plane
        config['n_clip_error'] = int(nbp_basic.tile_sz * nbp_basic.tile_sz / 100)

    with tqdm(total=n_images) as pbar:
        pbar.set_description(f'Loading in tiles from {nbp_file.raw_extension}, filtering and saving as .npy')
        for r in use_rounds:
            # set scale and channels to use
            im_file = os.path.join(nbp_file.input_dir, round_files[r])
            if nbp_file.raw_extension == '.npy':
                extract.wait_for_data(im_file, config['wait_time'], dir=True)
            else:
                extract.wait_for_data(im_file + nbp_file.raw_extension, config['wait_time'])
            round_dask_array = utils.raw.load(nbp_file, nbp_basic, r=r)
            if r == nbp_basic.anchor_round:
                n_clip_error_images = 0  # reset for anchor as different scale used.
                if config['scale_anchor'] is None:
                    nbp_debug.scale_anchor_tile, _, nbp_debug.scale_anchor_z, config['scale_anchor'] = \
                        extract.get_scale(nbp_file, nbp_basic, nbp_basic.anchor_round, nbp_basic.use_tiles,
                                          [nbp_basic.anchor_channel], nbp_basic.use_z,
                                          config['scale_norm'], filter_kernel, smooth_kernel_2d)
                    # save scale values incase need to re-run
                    extract.save_scale(nbp_file.scale, nbp_debug.scale, config['scale_anchor'])
                else:
                    nbp_debug.scale_anchor_tile = None
                    nbp_debug.scale_anchor_z = None
                nbp_debug.scale_anchor = config['scale_anchor']
                print(f"Scale_anchor: {nbp_debug.scale_anchor}")
                scale = nbp_debug.scale_anchor
                use_channels = use_channels_anchor
            else:
                scale = nbp_debug.scale
                use_channels = nbp_basic.use_channels

            # convolve_2d each image
            for t in nbp_basic.use_tiles:
                if not nbp_basic.is_3d:
                    # for 2d all channels in same file
                    file_exists = os.path.isfile(nbp_file.tile[t][r])
                    if file_exists:
                        # mmap load in image for all channels if tiff exists
                        im_all_channels_2d = np.load(nbp_file.tile[t][r], mmap_mode='r')
                    else:
                        # Only save 2d data when all channels collected
                        # For channels not used, keep all pixels 0.
                        im_all_channels_2d = np.zeros((nbp_basic.n_channels, nbp_basic.tile_sz,
                                                       nbp_basic.tile_sz), dtype=np.int32)
                for c in use_channels:
                    if r == nbp_basic.anchor_round and c == nbp_basic.anchor_channel:
                        # max value that can be saved and no shifting done for DAPI
                        max_npy_pixel_value = np.iinfo(np.uint16).max
                    else:
                        max_npy_pixel_value = np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift
                    if nbp_basic.is_3d:
                        file_exists = os.path.isfile(nbp_file.tile[t][r][c])
                    pbar.set_postfix({'round': r, 'tile': t, 'channel': c, 'exists': str(file_exists)})
                    if file_exists:
                        if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
                            pass
                        else:
                            # Only need to load in mid-z plane if 3D.
                            if nbp_basic.is_3d:
                                im = utils.npy.load_tile(nbp_file, nbp_basic, t, r, c,
                                                         yxz=[None, None, nbp_debug.z_info])
                            else:
                                im = im_all_channels_2d[c].astype(np.int32) - nbp_basic.tile_pixel_value_shift
                            nbp.auto_thresh[t, r, c], hist_counts_trc, nbp_debug.n_clip_pixels[t, r, c], \
                            nbp_debug.clip_extract_scale[t, r, c] = \
                                extract.get_extract_info(im, config['auto_thresh_multiplier'], hist_bin_edges,
                                                         max_npy_pixel_value, scale)
                            if r != nbp_basic.anchor_round:
                                nbp.hist_counts[:, r, c] += hist_counts_trc
                    else:
                        im = utils.raw.load(nbp_file, nbp_basic, round_dask_array, r, t, c, nbp_basic.use_z)
                        if not nbp_basic.is_3d:
                            im = extract.focus_stack(im)
                        im, bad_columns = extract.strip_hack(im)  # find faulty columns
                        if config['deconvolve']:
                            im = extract.wiener_deconvolve(im, config['wiener_pad_shape'], wiener_filter)
                        if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
                            im = utils.morphology.top_hat(im, filter_kernel_dapi)
                            im[:, bad_columns] = 0
                        else:
                            # im converted to float in convolve_2d so no point changing dtype beforehand.
                            im = utils.morphology.convolve_2d(im, filter_kernel) * scale
                            if config['r_smooth'] is not None:
                                # oa convolve uses lots of memory and much slower here.
                                im = utils.morphology.imfilter(im, smooth_kernel, oa=False)
                            im[:, bad_columns] = 0
                            # get_info is quicker on int32 so do this conversion first.
                            im = np.rint(im, np.zeros_like(im, dtype=np.int32), casting='unsafe')
                            # only use image unaffected by strip_hack to get information from tile
                            good_columns = np.setdiff1d(np.arange(nbp_basic.tile_sz), bad_columns)
                            nbp.auto_thresh[t, r, c], hist_counts_trc, nbp_debug.n_clip_pixels[t, r, c], \
                            nbp_debug.clip_extract_scale[t, r, c] = \
                                extract.get_extract_info(im[:, good_columns], config['auto_thresh_multiplier'],
                                                         hist_bin_edges, max_npy_pixel_value, scale,
                                                         nbp_debug.z_info)

                            # Deal with pixels outside uint16 range when saving
                            if nbp_debug.n_clip_pixels[t, r, c] > config['n_clip_warn']:
                                warnings.warn(f"\nTile {t}, round {r}, channel {c} has "
                                              f"{nbp_debug.n_clip_pixels[t, r, c]} pixels\n"
                                              f"that will be clipped when converting to uint16.")
                            if nbp_debug.n_clip_pixels[t, r, c] > config['n_clip_error']:
                                n_clip_error_images += 1
                                message = f"\nNumber of images for which more than {config['n_clip_error']} pixels " \
                                          f"clipped in conversion to uint16 is {n_clip_error_images}."
                                if n_clip_error_images >= config['n_clip_error_images_thresh']:
                                    # create new Notebook to save info obtained so far
                                    nb_fail_name = os.path.join(nbp_file.output_dir, 'notebook_extract_error.npz')
                                    nb_fail = Notebook(nb_fail_name, None)
                                    # change names of pages so can add extra properties not in json file.
                                    nbp.name = 'extract_fail'
                                    nbp_debug.name = 'extract_debug_fail'
                                    nbp.fail_trc = np.array([t, r, c])  # record where failure occurred
                                    nbp_debug.fail_trc = np.array([t, r, c])
                                    nb_fail += nbp
                                    nb_fail += nbp_debug
                                    raise ValueError(f"{message}\nResults up till now saved as {nb_fail_name}.")
                                else:
                                    warnings.warn(f"{message}\nWhen this reaches {config['n_clip_error_images_thresh']}"
                                                  f", the extract step of the algorithm will be interrupted.")

                            if r != nbp_basic.anchor_round:
                                nbp.hist_counts[:, r, c] += hist_counts_trc
                        if nbp_basic.is_3d:
                            utils.npy.save_tile(nbp_file, nbp_basic, im, t, r, c)
                        else:
                            im_all_channels_2d[c] = im
                    pbar.update(1)
                if not nbp_basic.is_3d:
                    utils.npy.save_tile(nbp_file, nbp_basic, im_all_channels_2d, t, r)
    pbar.close()
    if not nbp_basic.use_anchor:
        nbp_debug.scale_anchor_tile = None
        nbp_debug.scale_anchor_z = None
        nbp_debug.scale_anchor = None
    return nbp, nbp_debug
