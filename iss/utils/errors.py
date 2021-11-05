import numpy as np
import os
import re
from .tiff import load_tile_description


def out_of_bounds(var, var_values, min_allowed, max_allowed):
    """
    checks if all values i are within bounds

    :param var: string
        name of variable testing
    :param var_values: list or numpy array
    :param min_allowed: float or integer
    :param max_allowed: float or integer
    :return:
    """
    min_var = np.min(var_values)
    max_var = np.max(var_values)
    if min_var < min_allowed:
        raise ValueError(f"\nmin value of {var}, {min_var}, is below min allowed value: {min_allowed}")
    if max_var > max_allowed:
        raise ValueError(f"\nmax value of {var}, {max_var}, is above max allowed value: {max_allowed}")


def no_file(file_path):
    """
    raises error if file does not exist

    :param file_path: string, path to file of interest
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"\nNo file\n{file_path}\nexists")


def empty(var, var_values):
    """
    raises error if no data in var_values

    :param var: string
        name of variable testing
    :param var_values: list or numpy array
    """
    if len(var_values) == 0:
        raise ValueError(f"{var} contains no data")


def wrong_shape(var, var_values, expected_shape):
    """
    raises error if var_values shape is not expected_shape

    :param var: string
        name of variable testing
    :param var_values: numpy array
    :param expected_shape: list or numpy array
        shape var_values should be
    """
    actual_shape = np.array(var_values.shape)
    expected_shape = np.array(expected_shape)
    if len(actual_shape) != len(expected_shape):
        raise ValueError(f"Shape of {var} is {actual_shape} but should be {expected_shape}")
    elif np.abs(actual_shape - expected_shape).max() > 0:
        raise ValueError(f"Shape of {var} is {actual_shape} but should be {expected_shape}")


def check_tiff_description(nbp_file, nbp_basic, nbp_extract_params, t, c, r):
    """
    Check that scale in nbp_extract_params and tile_pixel_value_shift in nbp_basic
    match those used to make tiff files

    :param nbp_file: NotebookPage object containing file names
    :param nbp_basic: NotebookPage object containing basic info
    :param nbp_extract_params: NotebookPage object containing extract parameters
    :param t: integer, tiff tile index considering
    :param c: integer, channel considering
    :param r: integer, round considering
    """
    description = load_tile_description(nbp_file, nbp_basic, t, c, r)
    if "Scale = " in description and "Shift = " in description:
        # scale value is after 'Scale = ' in description
        scale_from_tiff = np.float64(description.split("Scale = ", 1)[1])
        # shift value is between 'Shift = ' and '. Scale = ' in description
        shift_from_tiff = int(re.findall(r'Shift = (.+?). Scale = ', description)[0])
        shift_from_log = nbp_basic['tile_pixel_value_shift']
        if r == nbp_basic['anchor_round'] and c == nbp_basic['anchor_channel']:
            scale_from_log = nbp_extract_params['scale_anchor']
        elif r != nbp_basic['anchor_round'] and c in nbp_basic['use_channels']:
            scale_from_log = nbp_extract_params['scale']
        else:
            scale_from_log = 1  # dapi image and un-used channels have no scaling
            shift_from_log = 0  # dapi image and un-used channels have no shift
        if scale_from_tiff != scale_from_log:
            raise ValueError(f"\nScale used to make tiff was {scale_from_tiff}."
                             f"\nBut current scale in log is {scale_from_log}.")
        if shift_from_tiff != shift_from_log:
            raise ValueError(f"\nShift used to make tiff was {shift_from_tiff}."
                             f"\nBut current tile_pixel_value_shift in log is {shift_from_log}.")
