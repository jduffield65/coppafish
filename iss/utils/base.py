from math import ceil, floor
import numpy as np


def round_any(x, base, round_type='round'):
    """
    rounds the number x to the nearest multiple of base with the rounding done according to round_type.
    e.g. round_any(3, 5) = 5. round_any(3, 5, 'floor') = 0.
    """
    if round_type == 'round':
        return base * round(x / base)
    elif round_type == 'ceil':
        return base * ceil(x / base)
    elif round_type == 'floor':
        return base * floor(x / base)


def setdiff2d(array1, array2):
    """
    finds all elements in array1 that are not in array2.
    Returned array will only contain unique elements e.g.
    if array1 has [4,0] twice, array2 has [4,0] once, returned array will not have [4,0].
    if array1 has [4,0] twice, array2 does not have [4,0], returned array will have [4,0] once.

    :param array1: numpy array [n_elements1 x element_dim]
    :param array2: numpy array [n_elements2 x element_dim]
    :return: numpy array [n_elements_diff x element_dim]
    """
    set1 = set([tuple(x) for x in array1])
    set2 = set([tuple(x) for x in array2])
    return np.array(list(set1-set2))
