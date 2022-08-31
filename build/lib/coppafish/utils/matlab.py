import h5py
import numpy as np
from scipy import io
from typing import Union, List
from ..setup.notebook import Notebook, NotebookPage
from ..call_spots.qual_check import quality_threshold


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


def update_dict(nbp: NotebookPage, nb: Notebook, spots_info_dict: dict, score_thresh: float) -> dict:
    """
    Used in `save_nb_results` to reduced amount of spots saved. Only spots with `score > score_thresh` kept.

    Args:
        nbp: Either `omp` or `ref_spots` NotebookPage.
        nb: Full notebook
        spots_info_dict: Dictionary produced in `save_nb_results` containing information to save about each spot.
        score_thresh: All spots with a score above this threshold will be returned.

    Returns:
        `spots_info_dict` is returned, just including spots for which `score > score_thresh`.

    """
    pf = nbp.name + '_'
    if pf != 'omp_' and pf != 'ref_spots_':
        raise ValueError("Wrong page given, should be 'omp' or 'ref_spots'")
    nbp.finalized = False
    score_thresh_old = nbp.score_thresh
    del nbp.score_thresh
    nbp.score_thresh = score_thresh
    if pf == 'omp_':
        keep = quality_threshold(nb, 'omp')
    else:
        keep = quality_threshold(nb, 'ref')
    del nbp.score_thresh
    nbp.score_thresh = score_thresh_old
    nbp.finalized = True
    for var in ['local_yxz', 'tile', 'colors', 'intensity', 'background_coef', 'gene_no']:
        spots_info_dict[pf + var] = spots_info_dict[pf + var][keep]
    if pf == 'ref_spots_':
        del spots_info_dict[pf + 'background_coef']
        for var in ['isolated', 'score', 'score_diff']:
            spots_info_dict[pf + var] = spots_info_dict[pf + var][keep]
    if pf == 'omp_':
        for var in ['shape_spot_local_yxz', 'shape_spot_gene_no', 'spot_shape_float']:
            del spots_info_dict[pf + var]
        for var in ['coef', 'n_neighbours_pos', 'n_neighbours_neg']:
            spots_info_dict[pf + var] = spots_info_dict[pf + var][keep]
    return spots_info_dict


def save_nb_results(nb: Notebook, file_name: str, score_thresh_ref_spots: float = 0.15, score_thresh_omp: float = 0.15):
    """
    Saves important information in notebook as a .mat file so can load in MATLAB and plot
    using python_testing/iss_object_from_python.m scripts

    Args:
        nb: Notebook containing at least `call_spots` page.
        file_name: Path to save matlab version of Notebook.
        score_thresh_ref_spots: Only `ref_round` spots with score exceeding this will be saved.
        score_thresh_omp: Only `omp` spots with score exceeding this will be saved.

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
        ref_spots_dict = update_dict(nb.ref_spots, nb, ref_spots_dict, score_thresh_ref_spots)
        mdic.update(ref_spots_dict)

    if nb.has_page('omp'):
        omp_dict = nb.omp.to_serial_dict()
        omp_dict = {omp_dict['PAGEINFO'] + '_' + k:v for k,v in omp_dict.items() if not '___' in k}
        del omp_dict['omp_PAGEINFO']
        omp_dict = update_dict(nb.omp, nb, omp_dict, score_thresh_omp)
        mdic.update(omp_dict)
    for k, v in mdic.items():
        if v is None:
            mdic[k] = []  # Can't save None values
    io.savemat(file_name, mdic)
