import tifffile
# import zarr
import numpy as np


def save(image, im_file, description=None, append=False):
    """
    save image as tiff at path given by im_file

    :param image: 2d (or 3d) numpy array [ny x nx (x nz or n_channels)]
    :param im_file: string. path to save file
    :param description: string, short description to save to metadata to describe image
        found after saving through tifffile.TiffFile(im_file).pages[0].tags["ImageDescription"].value
        default: None
    :param append: boolean. whether to add to file if it exists or replace, optional.
        default: False
    """
    # truncate image so don't get aliased values
    image[image > np.iinfo(np.uint16).max] = np.iinfo(np.uint16).max
    image[image < 0] = 0
    image = np.round(image).astype(np.uint16)
    if image.ndim == 3:
        # put dimension that is not y or x as first dimension so easier to load in a single plane later
        # and match MATLAB method of saving
        image = np.moveaxis(image, 2, 0)
    tifffile.imwrite(im_file, image, append=append, description=description)


def load(im_file, planes=None, y_roi=None, x_roi=None):
    """
    load tiff from path given by im_file

    :param im_file: string, path to tiff to load
    :param planes: integer or integer list/numpy array, optional.
        which planes in tiff file to load
        default: None meaning all planes
    :param y_roi: integer or integer list/numpy array, optional.
        y pixels to read in
        default: None meaning all y pixels
    :param x_roi: integer or integer list/numpy array
        x pixels to read in
        default: None meaning all x pixels
    :return: numpy array [len(y_roi) x len(x_roi) (x len(planes))]
    """
    image = tifffile.imread(im_file, key=planes)
    if image.ndim == 3:
        image = np.moveaxis(image, 0, 2)
    if y_roi is not None or x_roi is not None:
        if y_roi is None:
            y_roi = np.arange(image.shape[0])
        if x_roi is None:
            x_roi = np.arange(image.shape[1])
        y_roi = np.array([y_roi]).flatten().reshape(-1, 1)
        x_roi = np.array([x_roi]).flatten().reshape(1, -1).repeat(len(y_roi), 0)
        image = image[y_roi.reshape(-1, 1), x_roi]
    return image
