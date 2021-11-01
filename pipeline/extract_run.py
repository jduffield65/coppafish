import utils.nd2
import utils.errors
import extract
# from extract.base import *
# from extract.filter import get_pixel_length, hanning_diff, disk_strel
# from extract.scale import get_scale, select_tile
import numpy as np
import os
from tqdm import tqdm


def extract_and_filter(config, log_file, log_basic):
    """

    :param config:
    :param log_file:
    :param log_basic:
    :return:
    """
    '''initialise log object'''
    # initialise log object with info from config
    log_extract = config
    # initialise output of this part of pipeline as 'vars' key
    log_extract['vars'] = {'auto_thresh': np.zeros((log_basic['n_tiles'], log_basic['n_channels'],
                                                    log_basic['n_rounds'] + log_basic['n_extra_rounds'])),
                           'hist_values': np.arange(-log_basic['tile_pixel_value_shift'], np.iinfo(np.uint16).max -
                                                    log_basic['tile_pixel_value_shift'] + 2, 1)}
    log_extract['vars']['hist_counts'] = np.zeros((len(log_extract['vars']['hist_values']), log_basic['n_channels'],
                                                   log_basic['n_rounds'] + log_basic['n_extra_rounds']), dtype=int)
    hist_bin_edges = np.concatenate(
        (log_extract['vars']['hist_values'] - 0.5, log_extract['vars']['hist_values'][-1:] + 0.5))
    # initialise debugging info as 'diagnostic' key
    log_extract['diagnostic'] = {
        'n_clip_pixels': np.zeros_like(log_extract['vars']['auto_thresh'], dtype=int),
        'clip_extract_scale': np.zeros_like(log_extract['vars']['auto_thresh'])
    }

    '''update config params in log object'''
    if config['r1'] is None:
        log_extract['r1'] = extract.get_pixel_length(log_extract['r1_auto_microns'], log_basic['pixel_size']['xy'])
    if config['r2'] is None:
        log_extract['r2'] = log_extract['r1'] * 2
    if config['r_dapi'] is None:
        log_extract['r_dapi'] = extract.get_pixel_length(log_extract['r_dapi_auto_microns'],
                                                         log_basic['pixel_size']['xy'])

    filter_kernel = extract.hanning_diff(log_extract['r1'], log_extract['r2'])
    filter_kernel_dapi = extract.disk_strel(config['r_dapi'])

    if config['scale'] is None:
        # ensure scale_norm value is reasonable
        utils.errors.out_of_bounds('scale_norm + tile_pixel_value_shift',
                                   log_extract['scale_norm'] + log_basic['tile_pixel_value_shift'],
                                   log_basic['tile_pixel_value_shift'], np.iinfo('uint16').max)
        im_file = os.path.join(log_file['input_dir'], log_file['round'][0] + log_file['raw_extension'])
        log_extract['diagnostic']['scale_tile'], log_extract['diagnostic']['scale_channel'], \
        log_extract['diagnostic']['scale_z'], log_extract['scale'] = \
            extract.get_scale(im_file, log_basic['tilepos_yx'], log_basic['use_tiles'], log_basic['use_channels'],
                              log_basic['use_z'], log_extract['scale_norm'], filter_kernel)

    '''get rounds to iterate over'''
    use_channels_anchor = [c for c in [log_basic['anchor_channel'], log_basic['anchor_channel']] if c is not None]
    use_channels_anchor.sort()
    if log_basic['anchor_round'] is not None:
        # always have anchor as first round after imaging rounds
        round_files = log_file['round'] + [log_file['anchor']]
        use_rounds = log_basic['use_rounds'] + [log_basic['n_rounds']]
        n_images = (len(use_rounds) - 1) * len(log_basic['use_tiles']) * len(log_basic['use_channels']) + \
                   len(log_basic['use_tiles']) * len(use_channels_anchor)
    else:
        round_files = log_file['round']
        use_rounds = log_basic['use_rounds']
        n_images = len(use_rounds) * len(log_basic['use_tiles']) * len(log_basic['use_channels'])

    with tqdm(total=n_images) as pbar:
        for r in use_rounds:
            # set scale and channels to use
            im_file = os.path.join(log_file['input_dir'], round_files[r] + log_file['raw_extension'])
            extract.wait_for_data(im_file, config['wait_time'])
            images = utils.nd2.load(im_file)
            if r == log_basic['anchor_round']:
                if log_extract['scale_anchor'] is None:
                    log_extract['diagnostic']['scale_anchor_tile'], _, log_extract['diagnostic']['scale_anchor_z'], \
                    log_extract['scale_anchor'] = \
                        extract.get_scale(im_file, log_basic['tilepos_yx'],
                                          log_basic['use_tiles'], [log_basic['anchor_channel']], log_basic['use_z'],
                                          log_extract['scale_norm'], filter_kernel)
                scale = log_extract['scale_anchor']
                use_channels = use_channels_anchor
            else:
                scale = log_extract['scale']
                use_channels = log_basic['use_channels']

            # filter each image
            for t in log_basic['use_tiles']:
                if not log_basic['3d']:
                    # for 2d all channels in same file
                    file_exists = os.path.isfile(log_file['tile'][t, r])
                for c in range(log_basic['n_channels']):
                    if c in use_channels:
                        if log_basic['3d']:
                            file_exists = os.path.isfile(log_file['tile'][t, r, c])
                        if file_exists:
                            log_extract = extract.update_log_extract(log_file, log_basic, log_extract, hist_bin_edges,
                                                                     t, c, r)
                        else:
                            im = utils.nd2.get_image(images, extract.get_nd2_tile_ind(t, log_basic['tilepos_yx']),
                                                     c, log_basic['use_z'])
                            if not log_basic['3d']:
                                im = extract.focus_stack(im)
                            else:
                                im = im.astype(int)
                            im, bad_columns = extract.strip_hack(im)  # find faulty columns
                            if r == log_basic['anchor_round'] and c == log_basic['dapi_channel']:
                                im = extract.filter_dapi(im, filter_kernel_dapi)
                                im[:, bad_columns] = 0
                            else:
                                im = extract.filter_imaging(im, filter_kernel) * scale
                                im[:, bad_columns] = 0
                                im = np.round(im).astype(int)
                                log_extract = extract.update_log_extract(log_file, log_basic, log_extract,
                                                                         hist_bin_edges,
                                                                         t, c, r, im, bad_columns)
                            utils.tiff.save_tile(log_file, log_basic, log_extract, im, t, c, r)
                    elif not log_basic['3d'] and not file_exists:
                        # if not including channel, just set to all zeros
                        # only in 2D as all channels in same file - helps when loading in tiffs
                        im = np.zeros((log_basic['tile_sz'], log_basic['tile_sz']), dtype=np.uint16)
                        utils.tiff.save_tile(log_file, log_basic, log_extract, im, t, c, r)
                    pbar.update(1)
    pbar.close()
    return log_extract
