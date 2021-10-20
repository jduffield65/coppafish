import numpy as np
import os
import warnings
import time
from tqdm import tqdm
import utils.errors


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
        ['tiff']: numpy array[nTiles x 2] [i,:] contains YX position of tile with tiff index i.
    :return: integer, corresponding index in nd2 file.
    """
    return np.where(np.sum(tile_pos_yx['nd2'] == tile_pos_yx['tiff'][tile_ind_tiff], 1) == 2)[0][0]

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
