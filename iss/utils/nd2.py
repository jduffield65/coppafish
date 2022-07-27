import numpy as np
import nd2
import os
from . import errors
from typing import Optional, List, Union
import json
import numpy_indexed
import numbers


# bioformats ssl certificate error solution:
# https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3

# nd2 library Does not work with Mac M1
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

        - `xy_pos` - `List [n_tiles x 2]`. xy position of tiles in pixels.
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
    metadata['xy_pos'] = metadata['xy_pos'].tolist()
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


def save_metadata(json_file: str, nd2_file: str, use_channels: Optional[List] = None):
    """
    Saves the required metadata as a json file which will contain

    - `xy_pos` - `List [n_tiles x 2]`. xy position of tiles in pixels.
    - `pixel_microns` - `float`. xy pixel size in microns.
    - `pixel_microns_z` - `float`. z pixel size in microns.
    - `sizes` - dict with fov (`t`), channels (`c`), y, x, z-planes (`z`) dimensions.

    Args:
        json_file: Where to save json file
        nd2_file: Path to nd2 file
        use_channels: The channels which have been extracted from the nd2 file.
            If `None`, assume all channels in nd2 file used

    """
    metadata = get_metadata(nd2_file)
    if use_channels is not None:
        if len(use_channels) > metadata['sizes']['c']:
            raise ValueError(f"use_channels contains {len(use_channels)} channels but there "
                             f"are only {metadata['sizes']['c']} channels in the nd2 metadata.")
        metadata['sizes']['c'] = len(use_channels)
        metadata['use_channels'] = use_channels   # channels extracted from nd2 file
    json.dump(metadata, open(json_file, 'w'))


def get_nd2_tile_ind(tile_ind_npy: Union[int, List[int]], tile_pos_yx_nd2: np.ndarray,
                     tile_pos_yx_npy: np.ndarray) -> Union[int, List[int]]:
    """
    Gets index of tiles in nd2 file from tile index of npy file.

    Args:
        tile_ind_npy: Indices of tiles in npy file
        tile_pos_yx_nd2: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with nd2 index ```i```.
            Index 0 refers to ```YX = [0, 0]```.
            Index 1 refers to ```YX = [0, 1] if MaxX > 0```.
        tile_pos_yx_npy: ```int [n_tiles x 2]```.
            ```[i,:]``` contains YX position of tile with npy index ```i```.
            Index 0 refers to ```YX = [MaxY, MaxX]```.
            Index 1 refers to ```YX = [MaxY, MaxX - 1] if MaxX > 0```.

    Returns:
        Corresponding indices in nd2 file
    """
    if isinstance(tile_ind_npy, numbers.Number):
        tile_ind_npy = [tile_ind_npy]
    nd2_index = numpy_indexed.indices(tile_pos_yx_nd2, tile_pos_yx_npy[tile_ind_npy]).tolist()
    if len(nd2_index) == 1:
        return nd2_index[0]
    else:
        return nd2_index
    # return np.where(np.sum(tile_pos_yx_nd2 == tile_pos_yx_npy[tile_ind_npy], 1) == 2)[0][0]


# '''with nd2reader'''
# # Does not work with QuadCam data hence the switch to nd2 package
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
#         xy_pos: xy position of tiles in pixels. ([nTiles x 2] List)
#         pixel_microns: xy pixel size in microns (float)
#         pixel_microns_z: z pixel size in microns (float)
#         sizes: dictionary with fov (t), channels (c), y, x, z-planes (z) dimensions
#
#     :param file_name: path to desired nd2 file
#     """
#     images = load(file_name)
#     images = update_metadata(images)
#     full_metadata = images.metadata
#     metadata = {'sizes': full_metadata['sizes'],
#                 'pixel_microns': full_metadata['pixel_microns'],
#                 'pixel_microns_z': full_metadata['pixel_microns_z'],
#                 'xy_pos': full_metadata['xy_pos'].tolist()}
#     return metadata
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
#     :return: 3D uint16 numpy array
#     """
#     if use_z is None:
#         use_z = np.arange(images.sizes['z'])
#     else:
#         use_z = np.array(np.array(use_z).flatten())
#     image = np.zeros((images.sizes['x'], images.sizes['y'], len(use_z)), dtype=np.uint16)
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
