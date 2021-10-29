from nd2reader import ND2Reader
import numpy as np
import utils.errors

# bioformats ssl certificate error solution:
# https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3


def load(file_name):
    """
    :param file_name: path to desired nds2 file
    :return: ND2Reader object with z index
             iterating fastest and then channel index
             and then field of view.
    """
    utils.errors.no_file(file_name)
    images = ND2Reader(file_name)
    images.iter_axes = 'vcz'
    return images


def get_image(images, fov, channel, use_z=None):
    """
    get image as numpy array from nd2 file

    :param images: ND2Reader object with fov, channel, z as index order.
    :param fov: fov index of desired image
    :param channel: channel of desired image
    :param use_z: integer list, optional
        which z-planes of image to load
        default: will load all z-planes
    :return: 3D numpy array (float)
    """
    if use_z is None:
        use_z = np.arange(images.sizes['z'])
    image = np.zeros((images.sizes['x'], images.sizes['y'], len(use_z)), dtype=np.uint16)
    start_index = fov * images.sizes['c'] * images.sizes['z'] + channel * images.sizes['z']
    for i in range(len(use_z)):
        image[:, :, i] = images[start_index + use_z[i]]
    return image


def update_metadata(images):
    """
    Updates metadata dictionary in images to include:
    pixel_microns_z: z pixel size in microns (float)
    xy_pos: xy position of tiles in pixels. ([nTiles x 2] numpy array)

    :param images: ND2Reader object with metadata dictionary
    """
    if 'pixel_microns_z' not in images.metadata:
        # NOT 100% SURE THIS IS THE CORRECT VALUE!!
        images.metadata['pixel_microns_z'] = \
            images.parser._raw_metadata.image_calibration[b'SLxCalibration'][b'dAspect']
    if 'xy_pos' not in images.metadata:
        images.metadata['xy_pos'] = np.zeros((images.sizes['v'], 2))
        for i in range(images.sizes['v']):
            images.metadata['xy_pos'][i, 0] = images.parser._raw_metadata.x_data[i * images.sizes['z']]
            images.metadata['xy_pos'][i, 1] = images.parser._raw_metadata.y_data[i * images.sizes['z']]
        images.metadata['xy_pos'] = (images.metadata['xy_pos'] - np.min(images.metadata['xy_pos'], 0)
                                     ) / images.metadata['pixel_microns']
    return images
