import h5py
import numpy as np
from scipy import io
from typing import Union, List


def load_v_less_7_3(file_name: str, var_names: Union[str, List[str]]) -> Union[tuple, np.ndarray]:
    """
    This is used to load info from earlier than v7.3  matlab files.
    It is also good at dealing with complicated matlab cell arrays which are loaded as numpy object arrays.

    If `var_names` is `str`, one value is returned, otherwise tuple of all values requested is returned.

    Args:
        file_name: Path of MATLAB file.
        var_names: `str [n_vars]`.
            Names of variables desired.

    Returns:
        `Tuple` of `n_vars` numpy arrays.
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


def load_array(file_name: str, var_names: Union[str, List[str]]) -> Union[tuple, np.ndarray]:
    """
    This is used to load info from v7.3 or later matlab files.
    It is also good at dealing with complicated matlab cell arrays which are loaded as numpy object arrays.

    If `var_names` is `str`, one value is returned, otherwise `tuple` of all values requested is returned.

    Args:
        file_name: Path of MATLAB file.
        var_names: `str [n_vars]`.
            Names of variables desired.

    Returns:
        `Tuple` of `n_vars` numpy arrays.
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


def load_cell(file_name: str, var_name: str) -> list:
    """
    If cell is `M x N`, will return list of length `M` where each entry is another list of length `N`
    and each element of this list is a numpy array.

    Args:
        file_name: Path of MATLAB file.
        var_name: Names of variable in MATLAB file.

    Returns:
        MATLAB cell `var_name` as a list of numpy arrays.
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
