import numpy as np
import os
import warnings
import time
from tqdm import tqdm
import utils.errors
import utils.tiff


def wait_for_data(file_path, wait_time):
    """
    waits for wait_time seconds to see if file at file_path becomes available in that time.

    :param file_path: string, path to file of interest
    :param wait_time: integer, time to wait in seconds for file to become available.
    """
    if not os.path.isfile(file_path):
        # wait for file to become available
        if wait_time > 60 ** 2:
            wait_time_print = round(wait_time / 60 ** 2, 1)
            wait_time_unit = 'hours'
        else:
            wait_time_print = round(wait_time, 1)
            wait_time_unit = 'seconds'
        warnings.warn(f'\nNo file named\n{file_path}\nexists. Waiting for {wait_time_print} {wait_time_unit}...')
        for _ in tqdm(range(wait_time)):
            time.sleep(1)
            if os.path.isfile(file_path):
                break
        utils.errors.no_file(file_path)
        print("file found!\nWaiting for file to fully load...")
        # wait for file to stop loading
        old_bytes = 0
        new_bytes = 0.00001
        while new_bytes > old_bytes:
            time.sleep(5)
            old_bytes = new_bytes
            new_bytes = os.path.getsize(file_path)
        print("file loaded!")


def get_nd2_tile_ind(tile_ind_tiff, tile_pos_yx):
    """
    :param tile_ind_tiff: integer
        index of tiff file
    :param tile_pos_yx: dictionary
        ['nd2']: numpy array[nTiles x 2] [i,:] contains YX position of tile with nd2 index i.
            index -1 refers to YX = [0,0]
        ['tiff']: numpy array[nTiles x 2] [i,:] contains YX position of tile with tiff index i.
            index 0 refers to YX = [0,0]
    :return: integer, corresponding index in nd2 file.
    """
    return np.where(np.sum(tile_pos_yx['nd2'] == tile_pos_yx['tiff'][tile_ind_tiff], 1) == 2)[0][0]


def strip_hack(image):
    """
    finds all columns in image where each row is identical and then sets
    this column to the nearest normal column. Basically 'repeat padding'.

    :param image: numpy array (if 3d, last index is z)
        image from nd2 file, before filtering (can be after focus stacking)
    :return:
        image: numpy array, input array with change_columns set to nearest
        change_columns: numpy integer array, which columns have been changed
    """
    # all rows identical if standard deviation is 0
    if np.ndim(image) == 3:
        # assume each z-plane of 3d image has same bad columns
        # seems to always be the case for our data
        change_columns = np.where(np.std(image[:, :, 0], 0) == 0)[0]
    else:
        change_columns = np.where(np.std(image, 0) == 0)[0]
    good_columns = np.setdiff1d(np.arange(np.shape(image)[1]), change_columns)
    for col in change_columns:
        nearest_good_col = good_columns[np.argmin(np.abs(good_columns - col))]
        image[:, col] = image[:, nearest_good_col]
    return image, change_columns


def update_log_extract(log_file, log_basic, log_extract, hist_bin_edges, t, c, r,
                       image=None, bad_columns=None):
    """
    Calculate values for auto_thresh, hist_counts,
    n_clip_pixels and clip_extract_scale in log_extract.

    :param log_file: log object containing file names
    :param log_basic: log object containing basic info
    :param log_extract: log object containing extract info
    :param hist_bin_edges: numpy array [len(log_extract['vars']['hist_values']) + 1]
        hist_values shifted by 0.5 to give bin edges not centres.
    :param t: integer, tiff tile index considering
    :param c: integer, channel considering
    :param r: integer, round considering
    :param image: numpy int32 array [tile_sz x tile_sz (x nz)], optional
        default: None meaning image will be loaded in
    :param bad_columns: numpy integer array, optional.
        which columns of image have been changed after strip_hack
        default: None meaning strip_hack will be called
    :return: log_extract
    """
    if image is None:
        print(f"Round {r}, tile {t}, channel {c} already done.")
        file_exists = True
        image = utils.tiff.load_tile(log_file, log_basic, t, c, r, log_extract=log_extract)
    else:
        file_exists = False
    if bad_columns is None:
        _, bad_columns = strip_hack(image)

    # only use image unaffected by strip_hack to get information from tile
    good_columns = np.setdiff1d(np.arange(log_basic['tile_sz']), bad_columns)
    log_extract['vars']['auto_thresh'][t, c, r] = (np.median(np.abs(image[:, good_columns])) *
                                                   log_extract['auto_thresh_multiplier'])
    if r != log_basic['anchor_round']:
        log_extract['vars']['hist_counts'][:, c, r] += np.histogram(image[:, good_columns],
                                                                    hist_bin_edges)[0]
    if not file_exists:
        # if saving tile for first time, record how many pixels will be clipped
        # and a suitable scaling which would cause no clipping.
        # this part is never called for dapi so don't need to deal with exceptions
        max_tiff_pixel_value = np.iinfo(np.uint16).max - log_basic['tile_pixel_value_shift']
        n_clip_pixels = np.sum(image > max_tiff_pixel_value)
        log_extract['diagnostic']['n_clip_pixels'][t, c, r] = n_clip_pixels
        if n_clip_pixels > 0:
            if r == log_basic['anchor_round']:
                scale = log_extract['scale_anchor']
            else:
                scale = log_extract['scale']
            # image has already been multiplied by scale hence inclusion of scale here
            # max_tiff_pixel_value / image.max() is less than 1 so recommended scaling becomes smaller than scale.
            log_extract['diagnostic']['clip_extract_scale'][t, c, r] = scale * max_tiff_pixel_value / image.max()

    return log_extract
