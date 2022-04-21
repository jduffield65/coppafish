import numpy as np
from typing import Tuple, Union


class OutOfBoundsError(Exception):
    def __init__(self, var_name: str, oob_val: float, min_allowed: float, max_allowed: float):
        """
        Error raised because `oob_val` is outside expected range between
        `min_allowed` and `max_allowed` inclusive.

        Args:
            var_name: Name of variable testing.
            oob_val: Value in array that is not in expected range.
            min_allowed: Smallest allowed value i.e. `>= min_allowed`.
            max_allowed: Largest allowed value i.e. `<= max_allowed`.
        """
        self.message = f"\n{var_name} contains the value {oob_val}." \
                       f"\nThis is outside the expected inclusive range between {min_allowed} and {max_allowed}"
        super().__init__(self.message)


class NoFileError(Exception):
    def __init__(self, file_path: str):
        """
        Error raised because `file_path` does not exist.

        Args:
            file_path: Path to file of interest.
        """
        self.message = f"\nNo file with the following path:\n{file_path}\nexists"
        super().__init__(self.message)


class EmptyListError(Exception):
    def __init__(self, var_name: str):
        """
        Error raised because the variable indicated by `var_name` contains no data.

        Args:
            var_name: Name of list or numpy array
        """
        self.message = f"\n{var_name} contains no data"
        super().__init__(self.message)


def check_shape(array: np.ndarray, expected_shape: Union[list, tuple, np.ndarray]) -> bool:
    """
    Checks to see if `array` has the shape indicated by `expected_shape`.

    Args:
        array: Array to check the shape of.
        expected_shape: `int [n_array_dims]`.
            Expected shape of array.

    Returns:
        `True` if shape of array is correct.
    """
    correct_shape = array.ndim == len(expected_shape)  # first check if number of dimensions are correct
    if correct_shape:
        correct_shape = np.abs(np.array(array.shape) - np.array(expected_shape)).max() == 0
    return correct_shape


class ShapeError(Exception):
    def __init__(self, var_name: str, var_shape: tuple, expected_shape: tuple):
        """
        Error raised because variable indicated by `var_name` has wrong shape.

        Args:
            var_name: Name of numpy array.
            var_shape: Shape of numpy array.
            expected_shape: Expected shape of numpy array.
        """
        self.message = f"\nShape of {var_name} is {var_shape} but should be {expected_shape}"
        super().__init__(self.message)


class TiffError(Exception):
    def __init__(self, scale_tiff: float, scale_nbp: float, shift_tiff: int, shift_nbp: int):
        """
        Error raised because parameters used to produce tiff files are different to those in the current notebook.

        Args:
            scale_tiff: Scale factor applied to tiff. Found from tiff description.
            scale_nbp: Scale factor applied to tiff. Found from `nb.extract_debug.scale`.
            shift_tiff: Shift applied to tiff to ensure pixel values positive. Found from tiff description.
            shift_nbp: Shift applied to tiff to ensure pixel values positive.
                Found from `nb.basic_info.tile_pixel_value_shift`.
        """
        self.message = f"\nThere are differences between the parameters used to make the tiffs and the parameters " \
                       f"in the Notebook:"
        if scale_tiff != scale_nbp:
            self.message = self.message + f"\nScale used to make tiff was {scale_tiff}." \
                                          f"\nCurrent scale in extract_params notebook page is {scale_nbp}."
        if shift_tiff != shift_nbp:
            self.message = self.message + f"\nShift used to make tiff was {shift_tiff}." \
                                          f"\nCurrent tile_pixel_value_shift in basic_info notebook page is " \
                                          f"{shift_nbp}."
        super().__init__(self.message)
