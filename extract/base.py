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


def save_tiff(log_file, log_basic, log_extract, image, t, c, r):
    """
    wrapper function to save tiff files with correct shift and a short description in the metadata

    :param log_file: log object containing file names
    :param log_basic: log object containing basic info
    :param log_extract: log object containing extract info
    :param image: numpy float array [ny x nx (x nz)]
    :param t: integer, tiff tile index considering
    :param c: integer, channel considering
    :param r: integer, round considering
    """
    if r == log_basic['anchor_round']:
        round = "anchor"
        if c == log_basic['anchor_channel']:
            scale = log_extract['scale_anchor']
            shift = log_basic['tile_pixel_value_shift']
            channel = "anchor"
        elif c == log_basic['dapi_channel']:
            scale = 1
            shift = 0
            channel = "dapi"
        else:
            scale = 1
            shift = 0
            channel = "not used"
    else:
        round = r
        if c not in log_basic['use_channels']:
            scale = 1
            shift = 0
            channel = "not used"
        else:
            scale = log_extract['scale']
            shift = log_basic['tile_pixel_value_shift']
            channel = c
    description = f"Tile = {t}. Round = {round}. Channel = {channel}. Shift = {shift}. Scale = {scale}"
    image = image + shift
    if log_basic['3d']:
        utils.errors.wrong_shape('tile image', image, [log_basic['tile_sz'], log_basic['tile_sz'], log_basic['nz']])
        utils.tiff.save(image, log_file['tile'][t, r, c], append=False, description=description)
    else:
        utils.errors.wrong_shape('tile image', image, [log_basic['tile_sz'], log_basic['tile_sz']])
        utils.tiff.save(image, log_file['tile'][t, r], append=True, description=description)





# def get_filter(r1, r2):
#
# def fstack():
#
# def get_extract_scale():
#
# def filter():
#
# def filter_dapi():
#
# def get_hist_counts():
#
# def get_auto_thresh():
