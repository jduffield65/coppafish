import numpy as np


class OutOfBoundsError(Exception):
    def __init__(self, var_name, oob_val, min_allowed, max_allowed):
        """
        Error raised because oob_val is outside expected range between
        min_allowed and max_allowed inclusive.

        :param var_name: string, name of variable testing
        :param oob_val: float, value in array that is not in expected range
        :param min_allowed: float, smallest allowed value i.e. >= min_allowed
        :param max_allowed: float, largest allowed value i.e. <= max_allowed
        """
        self.message = f"\n{var_name} contains the value {oob_val}." \
                       f"\nThis is outside the expected range between {min_allowed} and {max_allowed}"
        super().__init__(self.message)


class NoFileError(Exception):
    def __init__(self, file_path):
        """
        Error raised because file_path does not exist

        :param file_path: string, path to file of interest
        """
        self.message = f"\nNo file with the following path:\n{file_path}\nexists"
        super().__init__(self.message)


class EmptyListError(Exception):
    def __init__(self, var_name):
        """
        Error raised because the variable indicated by var_name contains no data

        :param var_name: string, name of list or numpy array
        """
        self.message = f"\n{var_name} contains no data"
        super().__init__(self.message)


def check_shape(array, expected_shape):
    """
    Checks to see if array has the shape indicated by expected_shape.

    :param array: numpy array
    :param expected_shape: list, tuple or 1D numpy array [n_array_dims]
    :return: boolean, True if shape of array is correct
    """
    correct_shape = array.ndim == len(expected_shape)  # first check if number of dimensions are correct
    if correct_shape:
        correct_shape = np.abs(np.array(array.shape) - np.array(expected_shape)).max() == 0
    return correct_shape


class ShapeError(Exception):
    def __init__(self, var_name, var_shape, expected_shape):
        """
        Error raised because variable indicated by var_name has wrong shape

        :param var_name: string, name of numpy array
        :param var_shape: tuple, shape of numpy array
        :param expected_shape: tuple, expected shape of numpy array
        """
        self.message = f"\nShape of {var_name} is {var_shape} but should be {expected_shape}"
        super().__init__(self.message)


class TiffError(Exception):
    def __init__(self, scale_tiff, scale_nbp, shift_tiff, shift_nbp):
        """
        Error raised because parameters used to produce tiff files are different to those in the current notebook.

        :param scale_tiff: float, scale factor applied to tiff, found from tiff description
        :param scale_nbp: float, scale factor applied to tiff, found from nb['extract_params']['scale']
        :param shift_tiff: integer, shift applied to tiff to ensure pixel values positive.
            Found from tiff description
        :param shift_nbp: integer, shift applied to tiff to ensure pixel values positive.
            Found from nb['basic_info']['tile_pixel_value_shift']
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
