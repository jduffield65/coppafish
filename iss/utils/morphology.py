import numpy as np
import scipy.signal
from scipy.ndimage.morphology import grey_dilation
from scipy.ndimage import convolve, correlate
import numbers
from . import errors
import cv2


def ftrans2(b, t=None):
    """
    Produces a 2D convolve_2d that corresponds to the 1D convolve_2d b, using the transform t
    Copied from MATLAB ftrans2: https://www.mathworks.com/help/images/ref/ftrans2.html

    :param b: float numpy array [Q,]
    :param t: float numpy array [M, N], optional.
        default: McClellan transform
    :return: float numpy array [(M-1)*(Q-1)/2+1, (N-1)*(Q-1)/2+1]
    """
    if t is None:
        # McClellan transformation
        t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]]) / 8

    # Convert the 1-D convolve_2d b to SUM_n a(n) cos(wn) form
    n = int(round((len(b) - 1) / 2))
    b = b.reshape(-1, 1)
    b = np.rot90(np.fft.fftshift(np.rot90(b)))
    a = np.concatenate((b[:1], 2 * b[1:n + 1]))

    inset = np.floor((np.array(t.shape) - 1) / 2).astype(int)

    # Use Chebyshev polynomials to compute h
    p0 = 1
    p1 = t
    h = a[1] * p1
    rows = inset[0]
    cols = inset[1]
    h[rows, cols] += a[0] * p0
    for i in range(2, n + 1):
        p2 = 2 * scipy.signal.convolve2d(t, p1)
        rows = rows + inset[0]
        cols = cols + inset[1]
        p2[rows, cols] -= p0
        rows = inset[0] + np.arange(p1.shape[0])
        cols = (inset[1] + np.arange(p1.shape[1])).reshape(-1, 1)
        hh = h.copy()
        h = a[i] * p2
        h[rows, cols] += hh
        p0 = p1.copy()
        p1 = p2.copy()
    h = np.rot90(h)
    return h


def hanning_diff(r1, r2):
    """
    gets difference of two hanning window convolve_2d
    (central positive, outer negative) with sum of 0.

    :param r1: integer
        radius in pixels of central positive hanning convolve_2d
    :param r2: integer, must be greater than r1
        radius in pixels of outer negative hanning convolve_2d
    :return: float numpy array [2*r2 + 1, 2*r2 + 1]
    """
    errors.out_of_bounds('r1', r1, 0, r2)
    errors.out_of_bounds('r2', r2, r1, np.inf)
    h_outer = np.hanning(2 * r2 + 3)[1:-1]  # ignore zero values at first and last index
    h_outer = -h_outer / h_outer.sum()
    h_inner = np.hanning(2 * r1 + 3)[1:-1]
    h_inner = h_inner / h_inner.sum()
    h = h_outer.copy()
    h[r2 - r1:r2 + r1 + 1] += h_inner
    h = ftrans2(h)
    return h


def convolve_2d(image, kernel):
    """
    convolves image with kernel, padding by replicating border pixels
    np.flip is to give same as convn with replicate padding in MATLAB

    :param image: numpy array [image_sz1 x image_sz2]
    :param kernel: numpy float array
    :return: numpy float array [image_sz1 x image_sz2]
    """
    return cv2.filter2D(image.astype(float), -1, np.flip(kernel), borderType=cv2.BORDER_REPLICATE)


def ensure_odd_kernel(kernel, pad_location='start'):
    """
    This ensures all dimensions of kernel are odd by padding even dimensions with zeros.
    Replicates MATLAB way of dealing with even kernels.
    e.g. if pad_location is 'start': [[5,4];[3,1]] --> [[0,0,0],[0,5,4],[0,3,1]]

    :param kernel: numpy float array
        Multidimensional filter
    :param pad_location: string either 'start' or 'end'
        where to put zeros.
    :return: numpy float array
    """
    even_dims = (np.mod(kernel.shape, 2) == 0).astype(int)
    if max(even_dims) == 1:
        if pad_location == 'start':
            pad_dims = [tuple(np.array([1, 0]) * val) for val in even_dims]
        elif pad_location == 'end':
            pad_dims = [tuple(np.array([0, 1]) * val) for val in even_dims]
        else:
            raise ValueError(f"pad_location has to be either 'start' or 'end' but value given was {pad_location}.")
        return np.pad(kernel, pad_dims, mode='constant')
    else:
        return kernel


def top_hat(image, kernel):
    """
    does tophat filtering of image with kernel

    :param image: numpy float array [image_sz1 x image_sz2]
    :param kernel: numpy integer array containing only zeros or ones.
    :return: numpy float array [image_sz1 x image_sz2]
    """
    if np.max(np.mod(kernel.shape, 2) == 0):
        # With even kernel, gives different results to MATLAB
        raise ValueError(f'kernel dimensions are {kernel.shape}. Require all dimensions to be odd.')
    # kernel = ensure_odd_kernel(kernel)  # doesn't work for tophat at start or end.
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def dilate(image, kernel):
    """
    dilates image with kernel, using zero padding.

    :param image: numpy float array [image_sz1 x image_sz2]
    :param kernel: numpy integer array containing only zeros or ones.
    :return: numpy float array [image_sz1 x image_sz2]
    """
    kernel = ensure_odd_kernel(kernel)
    # mode refers to the padding. We pad with zeros to keep results the same as MATLAB
    return grey_dilation(image, footprint=kernel, mode='constant')
    # return morphology.dilation(image, kernel)


def imfilter(image, kernel, padding=0, corr_or_conv='corr'):
    """
    copy of MATLAB imfilter function with 'output_size' equal to 'same'.

    :param image: numpy float array [image_sz1 x image_sz2]
        Image to be filtered
    :param kernel: numpy float array
        Multidimensional filter
    :param padding:
        numeric scalar: Input array values outside the bounds of the array are assigned the value X.
                        When no padding option is specified, the default is 0.
        ‘reflect’: 	    Input array values outside the bounds of the array are computed by
                        mirror-reflecting the array across the array border.
        ‘nearest’:      Input array values outside the bounds of the array are assumed to equal
                        the nearest array border value.
        'wrap':         Input array values outside the bounds of the array are computed by implicitly
                        assuming the input array is periodic.
    :param corr_or_conv:
        'corr':         imfilter performs multidimensional filtering using correlation.
                        This is the default when no option specified.
        'conv':         imfilter performs multidimensional filtering using convolution.
    :return: numpy float array [image_sz1 x image_sz2]
    """
    if isinstance(padding, numbers.Number):
        pad_value = padding
        padding = 'constant'
    else:
        pad_value = 0.0  # doesn't do anything for non-constant padding
    if corr_or_conv == 'corr':
        kernel = ensure_odd_kernel(kernel, 'start')
        return correlate(image, kernel, mode=padding, cval=pad_value)
    elif corr_or_conv == 'conv':
        kernel = ensure_odd_kernel(kernel, 'end')
        return convolve(image, kernel, mode=padding, cval=pad_value)
    else:
        raise ValueError(f"corr_or_conv should be either 'corr' or 'conv' but given value is {corr_or_conv}")
