import numpy as np
import os


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