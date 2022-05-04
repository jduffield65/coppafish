import h5py
import numpy as np
from scipy import io
from typing import Union, List
from ..setup.notebook import Notebook


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


def save_nb_results(nb: Notebook, file_name: str):
    """
    Saves important information in notebook as a .mat file so can load in MATLAB and plot
    using python_testing/iss_object_from_python.m scripts

    Args:
        nb:
        file_name:
    """
    mdic = {}
    if nb.has_page('file_names'):
        mdic['tile_file_names'] = nb.file_names.tile
    if nb.has_page('stitch'):
        mdic['tile_origin'] = nb.stitch.tile_origin
    if nb.has_page('register'):
        mdic['transform'] = nb.register.transform

    if nb.has_page('call_spots'):
        call_spots_dict = nb.call_spots.to_serial_dict()
        call_spots_dict = {k: v for k, v in call_spots_dict.items() if not '___' in k}
        del call_spots_dict['PAGEINFO']
        mdic.update(call_spots_dict)

    if nb.has_page('ref_spots'):
        ref_spots_dict = nb.ref_spots.to_serial_dict()
        # Give 'ref_spots' prefix to key names as same variables in omp page.
        ref_spots_dict = {ref_spots_dict['PAGEINFO'] + '_' + k: v for k, v in ref_spots_dict.items() if not '___' in k}
        del ref_spots_dict['ref_spots_PAGEINFO']
        mdic.update(ref_spots_dict)

    if nb.has_page('omp'):
        omp_dict = nb.omp.to_serial_dict()
        omp_dict = {omp_dict['PAGEINFO'] + '_' + k:v for k,v in omp_dict.items() if not '___' in k}
        del omp_dict['omp_PAGEINFO']
        mdic.update(omp_dict)
    for k, v in mdic.items():
        if v is None:
            mdic[k] = []  # Can't save None values
    io.savemat(file_name, mdic)
