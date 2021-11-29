import iss.utils.strel
from iss import utils, extract
import iss.utils.morphology as morphology
# import iss.utils.errors
# from extract.base import *
# from extract.convolve_2d import get_pixel_length, hanning_diff, disk_strel
# from extract.scale import get_scale, select_tile
import numpy as np
import os
from tqdm import tqdm
from iss.setup.notebook import NotebookPage


def extract_and_filter(config, nbp_file, nbp_basic):
    """

    :param config:
    :param nbp_file:
    :param nbp_basic:
    :return:
    """
    '''initialise log object'''
    # initialise notebook pages
    nbp = NotebookPage("extract")
    nbp_params = NotebookPage("extract_params", config)  # params page inherits info from config
    nbp_debug = NotebookPage("extract_debug")
    # initialise output of this part of pipeline as 'vars' key
    nbp['auto_thresh'] = np.zeros((nbp_basic['n_tiles'], nbp_basic['n_channels'],
                                   nbp_basic['n_rounds'] + nbp_basic['n_extra_rounds']))
    nbp['hist_values'] = np.arange(-nbp_basic['tile_pixel_value_shift'], np.iinfo(np.uint16).max -
                                   nbp_basic['tile_pixel_value_shift'] + 2, 1)
    nbp['hist_counts'] = np.zeros((len(nbp['hist_values']), nbp_basic['n_channels'],
                                   nbp_basic['n_rounds'] + nbp_basic['n_extra_rounds']), dtype=int)
    hist_bin_edges = np.concatenate(
        (nbp['hist_values'] - 0.5, nbp['hist_values'][-1:] + 0.5))
    # initialise debugging info as 'debug' page
    nbp_debug['n_clip_pixels'] = np.zeros_like(nbp['auto_thresh'], dtype=int)
    nbp_debug['clip_extract_scale'] = np.zeros_like(nbp['auto_thresh'])

    '''update config params in log object'''
    if config['r1'] is None:
        nbp_params['r1'] = extract.get_pixel_length(config['r1_auto_microns'], nbp_basic['pixel_size_xy'])
    if config['r2'] is None:
        nbp_params['r2'] = nbp_params['r1'] * 2
    if config['r_dapi'] is None:
        nbp_params['r_dapi'] = extract.get_pixel_length(config['r_dapi_auto_microns'],
                                                        nbp_basic['pixel_size_xy'])
    filter_kernel = morphology.hanning_diff(nbp_params['r1'], nbp_params['r2'])
    filter_kernel_dapi = iss.utils.strel.disk(nbp_params['r_dapi'])

    if config['scale'] is None:
        # ensure scale_norm value is reasonable
        utils.errors.out_of_bounds('scale_norm + tile_pixel_value_shift',
                                   nbp_params['scale_norm'] + nbp_basic['tile_pixel_value_shift'],
                                   nbp_basic['tile_pixel_value_shift'], np.iinfo('uint16').max)
        im_file = os.path.join(nbp_file['input_dir'], nbp_file['round'][0] + nbp_file['raw_extension'])
        nbp_debug['scale_tile'], nbp_debug['scale_channel'], nbp_debug['scale_z'], nbp_params['scale'] = \
            extract.get_scale(im_file, nbp_basic['tilepos_yx'], nbp_basic['tilepos_yx_nd2'],
                              nbp_basic['use_tiles'], nbp_basic['use_channels'], nbp_basic['use_z'],
                              nbp_params['scale_norm'], filter_kernel)

    '''get rounds to iterate over'''
    use_channels_anchor = [c for c in [nbp_basic['dapi_channel'], nbp_basic['anchor_channel']] if c is not None]
    use_channels_anchor.sort()
    if nbp_basic['anchor_round'] is not None:
        # always have anchor as first round after imaging rounds
        round_files = nbp_file['round'] + [nbp_file['anchor']]
        use_rounds = nbp_basic['use_rounds'] + [nbp_basic['n_rounds']]
        n_images = (len(use_rounds) - 1) * len(nbp_basic['use_tiles']) * len(nbp_basic['use_channels']) + \
                   len(nbp_basic['use_tiles']) * len(use_channels_anchor)
    else:
        round_files = nbp_file['round']
        use_rounds = nbp_basic['use_rounds']
        n_images = len(use_rounds) * len(nbp_basic['use_tiles']) * len(nbp_basic['use_channels'])

    with tqdm(total=n_images) as pbar:
        for r in use_rounds:
            # set scale and channels to use
            im_file = os.path.join(nbp_file['input_dir'], round_files[r] + nbp_file['raw_extension'])
            extract.wait_for_data(im_file, config['wait_time'])
            images = utils.nd2.load(im_file)
            if r == nbp_basic['anchor_round']:
                if config['scale_anchor'] is None:
                    nbp_debug['scale_anchor_tile'], _, nbp_debug['scale_anchor_z'], nbp_params['scale_anchor'] = \
                        extract.get_scale(im_file, nbp_basic['tilepos_yx'], nbp['tilepos_yx_nd2'],
                                          nbp_basic['use_tiles'], [nbp_basic['anchor_channel']], nbp_basic['use_z'],
                                          nbp_params['scale_norm'], filter_kernel)
                scale = nbp_params['scale_anchor']
                use_channels = use_channels_anchor
            else:
                scale = nbp_params['scale']
                use_channels = nbp_basic['use_channels']

            # convolve_2d each image
            for t in nbp_basic['use_tiles']:
                if not nbp_basic['3d']:
                    # for 2d all channels in same file
                    file_exists = os.path.isfile(nbp_file['tile'][t][r])
                for c in range(nbp_basic['n_channels']):
                    if c in use_channels:
                        if nbp_basic['3d']:
                            file_exists = os.path.isfile(nbp_file['tile'][t][r][c])
                        pbar.set_postfix({'round': r, 'tile': t, 'channel': c, 'exists': str(file_exists)})
                        if file_exists:
                            nbp, nbp_debug = extract.update_log_extract(nbp_file, nbp_basic, nbp, nbp_params, nbp_debug,
                                                                        hist_bin_edges, t, c, r)
                        else:
                            im = utils.nd2.get_image(images, extract.get_nd2_tile_ind(t, nbp_basic['tilepos_yx_nd2'],
                                                                                      nbp_basic['tilepos_yx']),
                                                     c, nbp_basic['use_z'])
                            if not nbp_basic['3d']:
                                im = extract.focus_stack(im)
                            else:
                                im = im.astype(int)
                            im, bad_columns = extract.strip_hack(im)  # find faulty columns
                            if r == nbp_basic['anchor_round'] and c == nbp_basic['dapi_channel']:
                                im = morphology.top_hat(im, filter_kernel_dapi)
                                im[:, bad_columns] = 0
                            else:
                                im = morphology.convolve_2d(im, filter_kernel) * scale
                                im[:, bad_columns] = 0
                                im = np.round(im).astype(int)
                                nbp, nbp_debug = extract.update_log_extract(nbp_file, nbp_basic, nbp, nbp_params,
                                                                            nbp_debug, hist_bin_edges, t, c, r,
                                                                            im, bad_columns)
                            utils.tiff.save_tile(nbp_file, nbp_basic, nbp_params, im, t, c, r)
                    elif not nbp_basic['3d'] and not file_exists:
                        # if not including channel, just set to all zeros
                        # only in 2D as all channels in same file - helps when loading in tiffs
                        im = np.zeros((nbp_basic['tile_sz'], nbp_basic['tile_sz']), dtype=np.uint16)
                        utils.tiff.save_tile(nbp_file, nbp_basic, nbp_params, im, t, c, r)
                    pbar.update(1)
    pbar.close()
    return nbp, nbp_params, nbp_debug
