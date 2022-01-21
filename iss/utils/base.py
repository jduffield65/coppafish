from math import ceil, floor
import numpy as np


def round_any(x, base, round_type='round'):
    """
    rounds the x to the nearest multiple of base with the rounding done according to round_type.
    e.g. round_any(3, 5) = 5. round_any(3, 5, 'floor') = 0.

    :param x: float or numpy array
    :param base: float
    :param round_type: string, either 'round', 'ceil' or 'floor
        default: 'round'
    """
    if round_type == 'round':
        return base * np.round(x / base)
    elif round_type == 'ceil':
        return base * np.ceil(x / base)
    elif round_type == 'floor':
        return base * np.floor(x / base)
    else:
        raise ValueError(f"round_type specified was {round_type} but it should be one of the following:\n"
                         f"round, ceil, floor")


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


def multi_array_ind(*args):
    """
    returns the indices to get a sub array from a multi dimensional numpy array

    For example, if you have a numpy array of size [10, 3, 8] called big_array and you want to get the sub array
    from [4,8] of the first axis, [0,1,2] of the second axis and [0,4,6,7] of the third axis then you can run:
    ind1, ind2, ind3 = multi_array_ind([4,8], [0,1,2], [0,4,6,7]) and the sub array can be obtained from
    sub_array = big_array[ind1, ind2, ind3]

    :param args: provide n lists each containing the desired indices along an axis
    :return: returns a tuple containing n numpy arrays, each of shape [len(arg_1) x len(arg_2) x ... x len(arg_n)]
    """
    n_arrays = len(args)
    array_sizes = [len(array) for array in args]
    ind_order = np.ones(n_arrays, dtype=int)
    output = [[]] * n_arrays
    for i in range(n_arrays):
        ind_order_i = ind_order.copy()
        ind_order_i[i] = -1
        n_repeats = array_sizes.copy()
        n_repeats[i] = 1
        output[i] = np.tile(np.array(args[i]).reshape(ind_order_i), n_repeats)
    return tuple(output)
