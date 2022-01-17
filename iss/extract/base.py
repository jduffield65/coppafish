import numpy as np
import os
import warnings
import time
from tqdm import tqdm
from .. import utils


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
        if not os.path.isfile(file_path):
            raise utils.errors.NoFileError(file_path)
        print("file found!\nWaiting for file to fully load...")
        # wait for file to stop loading
        old_bytes = 0
        new_bytes = 0.00001
        while new_bytes > old_bytes:
            time.sleep(5)
            old_bytes = new_bytes
            new_bytes = os.path.getsize(file_path)
        print("file loaded!")


def get_pixel_length(length_microns, pixel_size):
    """
    Converts a length in units of microns into a length in units of pixels
    :param length_microns: float
        length in units of microns (microns)
    :param pixel_size: float
        size of a pixel in microns (microns/pixels)
    :return: integer, desired length in units of pixels (pixels)
    """
    return int(round(length_microns / pixel_size))


def get_nd2_tile_ind(tile_ind_tiff, tile_pos_yx_nd2, tile_pos_yx_tiff):
    """
    :param tile_ind_tiff: integer
        index of tiff file
    :param tile_pos_yx_nd2: numpy array[nTiles x 2]
        [i,:] contains YX position of tile with nd2 index i.
        index 0 refers to YX = [MaxY,MaxX]
    :param tile_pos_yx_tiff: numpy array[nTiles x 2]
        [i,:] contains YX position of tile with tiff index i.
        index 0 refers to YX = [0,0]
    :return: integer, corresponding index in nd2 file.
    """
    return np.where(np.sum(tile_pos_yx_nd2 == tile_pos_yx_tiff[tile_ind_tiff], 1) == 2)[0][0]


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


def update_log_extract(nbp_file, nbp_basic, nbp_vars, nbp_debug, auto_thresh_multiplier,
                       hist_bin_edges, t, r, c, image=None, bad_columns=None):
    """
    Calculate values for auto_thresh, hist_counts in nbp_vars and
    n_clip_pixels and clip_extract_scale in nbp_debug.

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param nbp_vars: NotebookPage object containing variables found during extract.
    :param nbp_debug: NotebookPage object containing debugging info found during extract.
    :param auto_thresh_multiplier: float
        auto_thresh is set to auto_thresh_multiplier * median(abs(image))
        so that pixel values above this are likely spots
        typical = 10
    :param hist_bin_edges: numpy array [len(log_extract['vars']['hist_values']) + 1]
        hist_values shifted by 0.5 to give bin edges not centres.
    :param t: integer, tiff tile index considering
    :param r: integer, round considering
    :param c: integer, channel considering
    :param image: numpy int32 array [tile_sz x tile_sz (x nz)], optional
        default: None meaning image will be loaded in
    :param bad_columns: numpy integer array, optional.
        which columns of image have been changed after strip_hack
        default: None meaning strip_hack will be called
    :return: nbp_extract, nbp_extract_debug
    """
    # TODO: I think this is very slow in 3d (30s per image when 50 z-plane image already in tile directory)
    if image is None:
        file_exists = True
        image = utils.tiff.load_tile(nbp_file, nbp_basic, t, r, c, nbp_extract_debug=nbp_debug)
    else:
        file_exists = False
    if bad_columns is None:
        _, bad_columns = strip_hack(image)

    # only use image unaffected by strip_hack to get information from tile
    good_columns = np.setdiff1d(np.arange(nbp_basic['tile_sz']), bad_columns)
    if not (r == nbp_basic['anchor_round'] and c == nbp_basic['dapi_channel']):
        nbp_vars['auto_thresh'][t, r, c] = (np.median(np.abs(image[:, good_columns])) *
                                            auto_thresh_multiplier)
    if r != nbp_basic['anchor_round']:
        nbp_vars['hist_counts'][:, r, c] += np.histogram(image[:, good_columns], hist_bin_edges)[0]
    if not file_exists:
        # if saving tile for first time, record how many pixels will be clipped
        # and a suitable scaling which would cause no clipping.
        # this part is never called for dapi so don't need to deal with exceptions
        max_tiff_pixel_value = np.iinfo(np.uint16).max - nbp_basic['tile_pixel_value_shift']
        n_clip_pixels = np.sum(image > max_tiff_pixel_value)
        nbp_debug['n_clip_pixels'][t, r, c] = n_clip_pixels
        if n_clip_pixels > 0:
            if r == nbp_basic['anchor_round']:
                scale = nbp_debug['scale_anchor']
            else:
                scale = nbp_debug['scale']
            # image has already been multiplied by scale hence inclusion of scale here
            # max_tiff_pixel_value / image.max() is less than 1 so recommended scaling becomes smaller than scale.
            nbp_debug['clip_extract_scale'][t, r, c] = scale * max_tiff_pixel_value / image.max()

    return nbp_vars, nbp_debug
