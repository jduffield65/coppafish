import numpy as np


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
        raise ValueError(f"min value of {var}, {min_var}, is below min allowed value: {min_allowed}")
    if max_var > max_allowed:
        raise ValueError(f"max value of {var}, {max_var}, is above max allowed value: {max_allowed}")
