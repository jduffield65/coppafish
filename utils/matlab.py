import h5py
import numpy as np


def load_array(file_name, var_name):
    f = h5py.File(file_name)
    return np.array(f[var_name]).transpose()


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
