import h5py
import numpy as np
from scipy import io


def load_v_less_7_3(file_name, var_names):
    """
    this is used to load info from earlier than v7.3  matlab files.
    it is also good at dealing with complicated matlab cell arrays which are loaded as numpy object arrays.

    :param file_name:
    :param var_names: string or list of strings
    :return:
    """
    f = io.loadmat(file_name)
    if not isinstance(var_names, list):
        output = f[var_names]
    else:
        output = []
        for var_name in var_names:
            output.append(f[var_name])
        output = tuple(output)
    return output


def load_array(file_name, var_names):
    """
    this is used to load info from v7.3 or later matlab

    :param file_name:
    :param var_names: string or list of strings
    :return:
    """
    f = h5py.File(file_name)
    if not isinstance(var_names, list):
        output = np.array(f[var_names]).transpose()
    else:
        output = []
        for var_name in var_names:
            output.append(np.array(f[var_name]).transpose())
        output = tuple(output)
    return output


def load_cell(file_name, var_name):
    """
    If cell is M x N, will return list of length M where each entry is another list of length N
    and each element of this list is a numpy array.
    :param file_name:
    :param var_name:
    :return:
    """
    # MAYBE CHANGE THIS TO OBJECT NUMPY ARRAY
    f = h5py.File(file_name)
    data = []
    for column in np.transpose(f[var_name]):
        row_data = []
        for row_number in range(len(column)):
            row_data.append(np.array(f[column[row_number]]).transpose())
        data.append(row_data)
    return data
