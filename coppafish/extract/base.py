import numpy as np
import os
import warnings
import time
from tqdm import tqdm
from .. import utils
from typing import Tuple, Optional


def wait_for_data(data_path: str, wait_time: int, dir: bool = False):
    """
    Waits for wait_time seconds to see if file/directory at data_path becomes available in that time.

    Args:
        data_path: Path to file or directory of interest
        wait_time: Time to wait in seconds for file to become available.
        dir: If True, assumes data_path points to a directory, otherwise assumes points to a file.
    """
    if dir:
        check_data_func = lambda x: os.path.isdir(x)
    else:
        check_data_func = lambda x: os.path.isfile(x)
    if not check_data_func(data_path):
        # wait for file to become available
        if wait_time > 60 ** 2:
            wait_time_print = round(wait_time / 60 ** 2, 1)
            wait_time_unit = 'hours'
        else:
            wait_time_print = round(wait_time, 1)
            wait_time_unit = 'seconds'
        warnings.warn(f'\nNo file named\n{data_path}\nexists. Waiting for {wait_time_print} {wait_time_unit}...')
        with tqdm(total=wait_time, position=0) as pbar:
            pbar.set_description(f"Waiting for {data_path}")
            for i in range(wait_time):
                time.sleep(1)
                if check_data_func(data_path):
                    break
                pbar.update(1)
        pbar.close()
        if not check_data_func(data_path):
            raise utils.errors.NoFileError(data_path)
        print("file found!\nWaiting for file to fully load...")
        # wait for file to stop loading
        old_bytes = 0
        new_bytes = 0.00001
        while new_bytes > old_bytes:
            time.sleep(5)
            old_bytes = new_bytes
            new_bytes = os.path.getsize(data_path)
        print("file loaded!")


def get_pixel_length(length_microns: float, pixel_size: float) -> int:
    """
    Converts a length in units of microns into a length in units of pixels

    Args:
        length_microns: Length in units of microns (microns)
        pixel_size: Size of a pixel in microns (microns/pixels)

    Returns:
        Desired length in units of pixels (pixels)

    """
    return int(round(length_microns / pixel_size))


def strip_hack(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds all columns in image where each row is identical and then sets
    this column to the nearest normal column. Basically 'repeat padding'.

    Args:
        image: ```float [n_y x n_x (x n_z)]```
            Image from nd2 file, before filtering (can be after focus stacking) and if 3d, last index must be z.

    Returns:
        - ```image``` - ```float [n_y x n_x (x n_z)]```
            Input array with change_columns set to nearest
        - ```change_columns``` - ```int [n_changed_columns]```
            Indicates which columns have been changed.
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


def get_extract_info(image: np.ndarray, auto_thresh_multiplier: float, hist_bin_edges: np.ndarray, max_pixel_value: int,
                     scale: float, z_info: Optional[int] = None) -> Tuple[float, np.ndarray, int, float]:
    """
    Gets information from filtered scaled images useful for later in the pipeline.
    If 3D image, only z-plane used for `auto_thresh` and `hist_counts` calculation for speed and the that the
    exact value of these is not that important, just want a rough idea.

    Args:
        image: ```int [n_y x n_x (x n_z)]```
            Image of tile after filtering and scaling.
        auto_thresh_multiplier: ```auto_thresh``` is set to ```auto_thresh_multiplier * median(abs(image))```
            so that pixel values above this are likely spots. Typical = 10
        hist_bin_edges: ```float [len(nbp['hist_values']) + 1]```
            ```hist_values``` shifted by 0.5 to give bin edges not centres.
        max_pixel_value: Maximum pixel value that image can contain when saving as tiff file.
            If no shift was applied, this would be ```np.iinfo(np.uint16).max```.
        scale: Factor by which, ```image``` has been multiplied in order to fill out available values in tiff file.
        z_info: z-plane to get `auto_thresh` and `hist_counts` from.

    Returns:
        - ```auto_thresh``` - ```int``` Pixel values above ```auto_thresh``` in ```image``` are likely spots.
        - ```hist_counts``` - ```int [len(nbp['hist_values'])]```.
            ```hist_counts[i]``` is the number of pixels found in ```image``` with value equal to
            ```hist_values[i]```.
        - ```n_clip_pixels``` - ```int``` Number of pixels in ```image``` with value more than ```max_pixel_value```.
        - ```clip_scale``` - ```float``` Suggested scale factor to multiply un-scaled ```image``` by in order for
            ```n_clip_pixels``` to be 0.
    """
    if image.ndim == 3:
        if z_info is None:
            raise ValueError("z_info not provided")
        auto_thresh = np.median(np.abs(image[:, :, z_info])) * auto_thresh_multiplier
        hist_counts = np.histogram(image[:, :, z_info], hist_bin_edges)[0]
    else:
        auto_thresh = np.median(np.abs(image)) * auto_thresh_multiplier
        hist_counts = np.histogram(image, hist_bin_edges)[0]
    n_clip_pixels = np.sum(image > max_pixel_value)
    if n_clip_pixels > 0:
        # image has already been multiplied by scale hence inclusion of scale here
        # max_pixel_value / image.max() is less than 1 so recommended scaling becomes smaller than scale.
        clip_scale = scale * max_pixel_value / image.max()
    else:
        clip_scale = 0
    return np.round(auto_thresh).astype(int), hist_counts, n_clip_pixels, clip_scale
