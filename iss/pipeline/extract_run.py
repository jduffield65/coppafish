from .. import utils, extract, setup
import numpy as np
import os
from tqdm import tqdm
from ..setup.notebook import NotebookPage
from typing import Tuple
import warnings


def extract_and_filter(config: dict, nbp_file: NotebookPage,
                       nbp_basic: NotebookPage) -> Tuple[NotebookPage, NotebookPage]:
    """
    This reads in images from the raw `nd2` files, filters them and then saves them as tiff files in the tile directory.
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
    # initialise notebook pages
    if not nbp_basic.is_3d:
        config['deconvolve'] = False  # only deconvolve if 3d pipeline
    nbp = setup.NotebookPage("extract")
    nbp_debug = setup.NotebookPage("extract_debug")
    # initialise output of this part of pipeline as 'vars' key
    nbp.auto_thresh = np.zeros((nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds,
                                nbp_basic.n_channels), dtype=int)
    nbp.hist_values = np.arange(-nbp_basic.tile_pixel_value_shift, np.iinfo(np.uint16).max -
                                nbp_basic.tile_pixel_value_shift + 2, 1)
    nbp.hist_counts = np.zeros((len(nbp.hist_values), nbp_basic.n_rounds, nbp_basic.n_channels), dtype=int)
    hist_bin_edges = np.concatenate((nbp.hist_values - 0.5, nbp.hist_values[-1:] + 0.5))
    # initialise debugging info as 'debug' page
    nbp_debug.n_clip_pixels = np.zeros_like(nbp.auto_thresh, dtype=int)
    nbp_debug.clip_extract_scale = np.zeros_like(nbp.auto_thresh)
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
        smooth_kernel = np.ones(tuple(np.array(config['r_smooth'], dtype=int)*2-1))
        smooth_kernel = smooth_kernel / np.sum(smooth_kernel)

        if np.max(config['r_smooth']) == 1:
            warnings.warn('Max radius of smooth filter was 1, so not using.')
            config['r_smooth'] = None

    if config['deconvolve']:
        if not os.path.isfile(nbp_file.psf):
            if nbp_basic.ref_round == nbp_basic.anchor_round:
                im_file = os.path.join(nbp_file.input_dir, nbp_file.anchor + nbp_file.raw_extension)
            else:
                im_file = os.path.join(nbp_file.input_dir,
                                       nbp_file.round[nbp_basic.ref_round] + nbp_file.raw_extension)

            spot_images, config['psf_intensity_thresh'], psf_tiles_used = \
                extract.get_psf_spots(im_file, nbp_basic.tilepos_yx, nbp_basic.tilepos_yx_nd2,
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

    if config['scale'] is None:
        # ensure scale_norm value is reasonable so max pixel value in tiff file is significant factor of max of uint16
        scale_norm_min = (np.iinfo('uint16').max - nbp_basic.tile_pixel_value_shift) / 5
        scale_norm_max = np.iinfo('uint16').max - nbp_basic.tile_pixel_value_shift
        if not scale_norm_min <= config['scale_norm'] <= scale_norm_max:
            raise utils.errors.OutOfBoundsError("scale_norm", config['scale_norm'], scale_norm_min, scale_norm_max)
        im_file = os.path.join(nbp_file.input_dir, nbp_file.round[0] + nbp_file.raw_extension)
        # If using smoothing, apply this as well before scal
        if config['r_smooth'] is None:
            smooth_kernel_2d = None
        else:
            if smooth_kernel.ndim == 3:
                # take central plane of smooth filter if 3D as scale is found from a single z-plane.
                smooth_kernel_2d = smooth_kernel[:, :, config['r_smooth'][2]-1]
            else:
                smooth_kernel_2d = smooth_kernel.copy()
            # smoothing is averaging so to average in 2D, need to re normalise filter
            smooth_kernel_2d = smooth_kernel_2d / np.sum(smooth_kernel_2d)
        nbp_debug.scale_tile, nbp_debug.scale_channel, nbp_debug.scale_z, config['scale'] = \
            extract.get_scale(im_file, nbp_basic.tilepos_yx, nbp_basic.tilepos_yx_nd2,
                              nbp_basic.use_tiles, nbp_basic.use_channels, nbp_basic.use_z,
                              config['scale_norm'], filter_kernel, smooth_kernel_2d)
    else:
        nbp_debug.scale_tile = None
        nbp_debug.scale_channel = None
        nbp_debug.scale_z = None
        smooth_kernel_2d = None
    nbp_debug.scale = config['scale']

    '''get rounds to iterate over'''
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

    with tqdm(total=n_images) as pbar:
        pbar.set_description('Loading in tiles from nd2, filtering and saving as npy')
        for r in use_rounds:
            # set scale and channels to use
            im_file = os.path.join(nbp_file.input_dir, round_files[r] + nbp_file.raw_extension)
            extract.wait_for_data(im_file, config['wait_time'])
            images = utils.nd2.load(im_file)
            if r == nbp_basic.anchor_round:
                if config['scale_anchor'] is None:
                    nbp_debug.scale_anchor_tile, _, nbp_debug.scale_anchor_z, config['scale_anchor'] = \
                        extract.get_scale(im_file, nbp_basic.tilepos_yx, nbp_basic.tilepos_yx_nd2,
                                          nbp_basic.use_tiles, [nbp_basic.anchor_channel], nbp_basic.use_z,
                                          config['scale_norm'], filter_kernel, smooth_kernel_2d)
                else:
                    nbp_debug.scale_anchor_tile = None
                    nbp_debug.scale_anchor_z = None
                nbp_debug.scale_anchor = config['scale_anchor']
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
                        max_tiff_pixel_value = np.iinfo(np.uint16).max
                    else:
                        max_tiff_pixel_value = np.iinfo(np.uint16).max - nbp_basic.tile_pixel_value_shift
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
                                                         max_tiff_pixel_value, scale)
                            if r != nbp_basic.anchor_round:
                                nbp.hist_counts[:, r, c] += hist_counts_trc
                    else:
                        im = utils.nd2.get_image(images, extract.get_nd2_tile_ind(t, nbp_basic.tilepos_yx_nd2,
                                                                                  nbp_basic.tilepos_yx),
                                                 c, nbp_basic.use_z)
                        if not nbp_basic.is_3d:
                            im = extract.focus_stack(im)
                        im, bad_columns = extract.strip_hack(im)  # find faulty columns
                        if config['deconvolve']:
                            im = extract.wiener_deconvolve(im, config['wiener_pad_shape'], wiener_filter)
                        if r == nbp_basic.anchor_round and c == nbp_basic.dapi_channel:
                            im = utils.morphology.top_hat(im, filter_kernel_dapi)
                            im[:, bad_columns] = 0
                        else:
                            # im converted to float in convolve_2d so no point changing dtype before hand.
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
                                                         hist_bin_edges, max_tiff_pixel_value, scale,
                                                         nbp_debug.z_info)
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
