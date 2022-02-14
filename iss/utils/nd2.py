import numpy as np
import nd2
import os
from . import errors
from typing import Optional, List

# bioformats ssl certificate error solution:
# https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3

# TODO: nd2 does not work on new macboook


def load(file_path: str) -> np.ndarray:
    """
    Returns dask array with indices in order `fov`, `channel`, `y`, `x`, `z`.

    Args:
        file_path: Path to desired nd2 file.

    Returns:
        Dask array indices in order `fov`, `channel`, `y`, `x`, `z`.
    """
    if not os.path.isfile(file_path):
        raise errors.NoFileError(file_path)
    images = nd2.ND2File(file_path)
    images = images.to_dask()
    # images = nd2.imread(file_name, dask=True)  # get python crashing with this in get_image for some reason
    images = np.moveaxis(images, 1, -1)  # put z index to end
    return images


def get_metadata(file_path: str) -> dict:
    """
    Gets metadata containing information from nd2 data about pixel sizes, position of tiles and numbers of
    tiles/channels/z-planes.

    Args:
        file_path: Path to desired nd2 file.

    Returns:
        Dictionary containing -

        - `xy_pos` - `np.ndarray [n_tiles x 2]`. xy position of tiles in pixels.
        - `pixel_microns` - `float`. xy pixel size in microns.
        - `pixel_microns_z` - `float`. z pixel size in microns.
        - `sizes` - dict with fov (`t`), channels (`c`), y, x, z-planes (`z`) dimensions.
    """
    if not os.path.isfile(file_path):
        raise errors.NoFileError(file_path)
    images = nd2.ND2File(file_path)
    metadata = {'sizes': {'t': images.sizes['P'], 'c': images.sizes['C'], 'y': images.sizes['Y'],
                          'x': images.sizes['X'], 'z': images.sizes['Z']},
                'pixel_microns': images.metadata.channels[0].volume.axesCalibration[0],
                'pixel_microns_z': images.metadata.channels[0].volume.axesCalibration[2]}
    xy_pos = np.array([images.experiment[0].parameters.points[i].stagePositionUm[:2]
                       for i in range(images.sizes['P'])])
    metadata['xy_pos'] = (xy_pos - np.min(xy_pos, 0)) / metadata['pixel_microns']
    return metadata


def get_image(images: np.ndarray, fov: int, channel: int, use_z: Optional[List[int]] = None) -> np.ndarray:
    """
    Using dask array from nd2 file, this loads the image of the desired fov and channel.

    Args:
        images: Dask array with `fov`, `channel`, y, x, z as index order.
        fov: `fov` index of desired image
        channel: `channel` of desired image
        use_z: `int [n_use_z]`.
            Which z-planes of image to load.
            If `None`, will load all z-planes.

    Returns:
        `uint16 [im_sz_y x im_sz_x x n_use_z]`.
            Image of the desired `fov` and `channel`.
    """
    if use_z is None:
        use_z = np.arange(images.shape[-1])
    return np.asarray(images[fov, channel, :, :, use_z])



'''with nd2reader'''
# from nd2reader import ND2Reader
#
#
# def load(file_path):
#     """
#     :param file_path: path to desired nd2 file
#     :return: ND2Reader object with z index
#              iterating fastest and then channel index
#              and then field of view.
#     """
#     if not os.path.isfile(file_path):
#         raise errors.NoFileError(file_path)
#     images = ND2Reader(file_path)
#     images.iter_axes = 'vcz'
#     return images
#
#
# def get_metadata(file_name):
#     """
#     returns dictionary containing (at the bare minimum) the keys
#         xy_pos: xy position of tiles in pixels. ([nTiles x 2] numpy array)
#         pixel_microns: xy pixel size in microns (float)
#         pixel_microns_z: z pixel size in microns (float)
#         sizes: dictionary with fov (t), channels (c), y, x, z-planes (z) dimensions
#
#     :param file_name: path to desired nd2 file
#     """
#     images = load(file_name)
#     images = update_metadata(images)
#     return images.metadata
#
#
# def get_image(images, fov, channel, use_z=None):
#     """
#     get image as numpy array from nd2 file
#
#     :param images: ND2Reader object with fov, channel, z as index order.
#     :param fov: fov index of desired image
#     :param channel: channel of desired image
#     :param use_z: integer list, optional
#         which z-planes of image to load
#         default: will load all z-planes
#     :return: 3D numpy array
#     """
#     if use_z is None:
#         use_z = np.arange(images.sizes['z'])
#     image = np.zeros((images.sizes['x'], images.sizes['y'], len(np.array(np.array(use_z).flatten()))), dtype=np.uint16)
#     start_index = fov * images.sizes['c'] * images.sizes['z'] + channel * images.sizes['z']
#     for i in range(len(use_z)):
#         image[:, :, i] = images[start_index + use_z[i]]
#     return image
#
#
# def update_metadata(images):
#     """
#     Updates metadata dictionary in images to include:
#     pixel_microns_z: z pixel size in microns (float)
#     xy_pos: xy position of tiles in pixels. ([nTiles x 2] numpy array)
#     sizes: dictionary with fov (t), channels (c), y, x, z-planes (z) dimensions
#
#     :param images: ND2Reader object with metadata dictionary
#     """
#     if 'pixel_microns_z' not in images.metadata:
#         # NOT 100% SURE THIS IS THE CORRECT VALUE!!
#         images.metadata['pixel_microns_z'] = \
#             images.parser._raw_metadata.image_calibration[b'SLxCalibration'][b'dAspect']
#     if 'xy_pos' not in images.metadata:
#         images.metadata['xy_pos'] = np.zeros((images.sizes['v'], 2))
#         for i in range(images.sizes['v']):
#             images.metadata['xy_pos'][i, 0] = images.parser._raw_metadata.x_data[i * images.sizes['z']]
#             images.metadata['xy_pos'][i, 1] = images.parser._raw_metadata.y_data[i * images.sizes['z']]
#         images.metadata['xy_pos'] = (images.metadata['xy_pos'] - np.min(images.metadata['xy_pos'], 0)
#                                      ) / images.metadata['pixel_microns']
#     if 'sizes' not in images.metadata:
#         images.metadata['sizes'] = {'t': images.sizes['v'], 'c': images.sizes['c'], 'y': images.sizes['y'],
#                                     'x': images.sizes['x'], 'z': images.sizes['z']}
#     return images
