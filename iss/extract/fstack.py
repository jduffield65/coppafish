import numpy as np
from scipy import ndimage
from scipy.signal import medfilt2d
import cv2


def rgb2gray(im):
    """
    Converts RGB m x n x 3 color image into m x n greyscale image
    Uses weighted sum indicated here:
    https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    :param im: numpy array [M x N x 3]
    :return: numpy array [M x N]
    """
    im_R = im[:, :, 0]
    im_G = im[:, :, 1]
    im_B = im[:, :, 2]
    if np.issubdtype(im.dtype,np.integer):
        im_output = np.round((0.2989 * im_R + 0.5870 * im_G + 0.1140 * im_B)).astype(im.dtype)
    else:
        im_output = (0.2989 * im_R + 0.5870 * im_G + 0.1140 * im_B).astype(im.dtype)
    return im_output


def im2double(im):
    """
    Equivalent to Matlab im2double function
    Follows answer here:
    https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python/29104511

    :param im: int or uint numpy array
    :return:
    """
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return im.astype(float) / info.max  # Divide all values by the largest possible value in the datatype


def focus_stack(im_stack, nhsize=9, focus=None, alpha=0.2, sth=13):
    """
    modified by Josh, 2021
    Focus stacking.

    SINTAX:
      im = focus_stack(imlist)
      im = focus_stack(imlist, opt1=val1, opt2=val2,...)

    DESCRIPTION:
    Generate extended depth-of-field image from focus sequence
    using noise-robust selective all-in-focus algorithm [1].
    Input images may be grayscale or color. For color images,
    the algorithm is applied to each color plane independently

    For further details, see:
    [1] Pertuz et. al. "Generation of all-in-focus images by
      noise-robust selective fusion of limited depth-of-field
      images" IEEE Trans. Image Process, 22(3):1242 - 1251, 2013.

    S. Pertuz
    Jan/2016


    :param im_stack: numpy array
        Element [:,:,p,:] is greyscale or RGB image at z-plane p.
        RGB: [M x N x P x 3] uint8
        Gray: [M x N x P] uint16
    :param nhsize: integer, optional.
        Size of focus measure window
        default: 9
    :param focus: numpy array [P,], optional.
        Vector with the focus of each frame.
        default: 0:len(imlist)
    :param alpha: float, optional.
        A scalar in [0,1]. See [1] for details.
        default: 0.2
    :param sth:  float, optional.
        A scalar. See [1] for details.
        default: 13
    :return:
        im is a MxN matrix with the all-in-focus (AIF) image.
    """
    rgb = np.ndim(im_stack) == 4
    if focus is None:
        focus = np.arange(im_stack.shape[2])
    if rgb:
        imagesR = im_stack[:, :, :, 0].astype(float)
        imagesG = im_stack[:, :, :, 1].astype(float)
        imagesB = im_stack[:, :, :, 2].astype(float)
    fm = get_fmeasure(im_stack, nhsize)
    S, fm = get_smeasure(fm, nhsize, focus)
    fm = get_weights(S, fm, alpha, sth)
    '''Fuse Images'''
    fmn = np.sum(fm, 2)  # normalisation factor
    if rgb:
        im = np.zeros_like(im_stack[:, :, 0])
        im[:, :, 0] = np.sum(imagesR * fm, 2) / fmn
        im[:, :, 1] = np.sum(imagesG * fm, 2) / fmn
        im[:, :, 2] = np.sum(imagesB * fm, 2) / fmn
        im = np.round(im).astype(np.uint8)
    else:
        im = np.round((np.sum(im_stack.astype(float) * fm, 2) / fmn)).astype(int)
    return im


def get_fmeasure(im_stack, nhsize):
    """
    Returns focus measure value for each pixel

    :param im_stack: numpy array
        Element [:,:,p,:] is greyscale or RGB image at z-plane p.
        RGB: [M x N x P x 3] uint8
        Gray: [M x N x P] uint16
    :param nhsize: integer, optional.
        Size of focus measure window
        typical: M/200
    :return:
        fm: numpy array [M x N x P]
    """
    rgb = np.ndim(im_stack) == 4
    im_shape = np.shape(im_stack)
    fm = np.zeros((im_shape[:3]))
    for p in range(im_shape[2]):
        if rgb:
            im = rgb2gray(im_stack[:, :, p])
        else:
            im = im_stack[:, :, p]
        fm[:, :, p] = gfocus(im2double(im), nhsize)
    return fm


def get_smeasure(fm, nhsize, focus):
    """
    Returns selectivity measure value for each pixel

    :param fm: numpy array [M x N x P]
        focus measure
    :param nhsize: integer
        Size of focus measure window
        typical: M/200
    :param focus: numpy array [P,]
        Vector with the focus of each frame.
        typical: 0:P
    :return:
        S: numpy array [M x N]
        fm: numpy array [M x N x P]

    """
    M, N, P = np.shape(fm)
    u, s_squared, A, fmax = gauss3p(focus, fm)
    # Aprox. RMS of error signal as sum|Signal-Noise| instead of sqrt(sum(Signal-noise)^2):
    err = np.zeros((M, N))
    for p in range(P):
        # +1 for self.focus is due to python vs matlab indexing
        # this bit is twice as slow vectorized for 2048 x 2048 x 51 image
        err = err + np.abs(fm[:, :, p] - A * np.exp(-(focus[p] + 1 - u) ** 2 / (2 * s_squared)))
    fm = fm / np.expand_dims(fmax, 2)
    h = np.ones((nhsize, nhsize)) / (nhsize ** 2)
    inv_psnr = ndimage.correlate(err / (P * fmax), h, mode='nearest')
    # inv_psnr = cv2.filter2D(err / (P * fmax), -1, h, borderType=cv2.BORDER_REPLICATE)
    S = 20 * np.log10(1 / inv_psnr)
    S[np.isnan(S)] = np.nanmin(S)
    return S, fm


def get_weights(S, fm, alpha, sth):
    """
    Computes sharpening parameter phi and then
    returns cut off frequency for high pass filter, omega.

    :param S: numpy array [M x N]
        selectivity measure
    :param fm: numpy array [M x N x P]
        normalised focus measure
    :param alpha: float.
        A scalar in [0,1].
        typical: 0.2
    :param sth: float.
        A scalar.
        typical: 13
    :return:
        omega: numpy array [M x N]
    """
    phi = 0.5 * (1 + np.tanh(alpha * (S - sth))) / alpha
    phi = medfilt2d(phi, 3)
    omega = 0.5 + 0.5 * np.tanh(np.expand_dims(phi, 2) * (fm - 1))
    return omega


def gfocus(im, w_size):
    """
    Compute focus measure using gray level local variance

    :param im: numpy array [M x N]
        gray scale image
    :param w_size: integer.
        Size of filter window
        typical: M/200
    :return:
        fm: numpy array [M x N]
    """
    mean_f = np.ones((w_size, w_size)) / (w_size ** 2)
    #u = ndimage.correlate(im, mean_f, mode='nearest')
    u = cv2.filter2D(im, -1, mean_f, borderType=cv2.BORDER_REPLICATE)
    fm = (im - u) ** 2
    fm = cv2.filter2D(fm, -1, mean_f, borderType=cv2.BORDER_REPLICATE)
    #fm = ndimage.correlate(fm, mean_f, mode='nearest')
    return fm


def gauss3p(x, y):
    """
    Fast 3-point gaussian interpolation

    :param x: numpy array [P,]
        Vector with the focus of each frame.
        typical: 0:P
    :param y: numpy array [M x N x P]
    :return:
        u: numpy array [M x N], mean value of gaussian function
        s_squared: numpy array [M x N], variance.
        A: numpy array [M x N], max value of gaussian function.
        y_max: numpy array [M x N], max projection of image y.

    """
    step = 2  # internal parameter
    M, N, P = np.shape(y)
    y_max = np.max(y, axis=2)
    I = np.argmax(y, axis=2)
    IN, IM = np.meshgrid(np.arange(N), np.arange(M))
    Ic = I.flatten()  # transpose before FLATTEN to give same as MATLAB
    Ic[Ic <= step - 1] = step
    Ic[Ic >= P - 1 - step] = P - 1 - step
    index1 = np.ravel_multi_index((IM.flatten(), IN.flatten(), Ic - step), (M, N, P))
    index2 = np.ravel_multi_index((IM.flatten(), IN.flatten(), Ic), (M, N, P))
    index3 = np.ravel_multi_index((IM.flatten(), IN.flatten(), Ic + step), (M, N, P))
    index1[I.flatten() <= step - 1] = index3[I.flatten() <= step - 1]
    index3[I.flatten() >= step - 1] = index1[I.flatten() >= step - 1]
    x1 = x[Ic.flatten() - step].reshape(M, N)
    x2 = x[Ic.flatten()].reshape(M, N)
    x3 = x[Ic.flatten() + step].reshape(M, N)
    y1 = np.log(y[np.unravel_index(index1, np.shape(y))].reshape(M, N))
    y2 = np.log(y[np.unravel_index(index2, np.shape(y))].reshape(M, N))
    y3 = np.log(y[np.unravel_index(index3, np.shape(y))].reshape(M, N))
    c = ((y1 - y2) * (x2 - x3) - (y2 - y3) * (x1 - x2)) / (
            (x1 ** 2 - x2 ** 2) * (x2 - x3) - (x2 ** 2 - x3 ** 2) * (x1 - x2))
    b = ((y2 - y3) - c * (x2 - x3) * (x2 + x3 + 2)) / (x2 - x3)  # +2 is due to python vs matlab indexing
    s_squared = -1 / (2 * c)
    u = b * s_squared
    a = y1 - b * (x1 + 1) - c * (x1 + 1) ** 2  # +1 is due to python vs matlab indexing
    A = np.exp(a + u ** 2. / (2 * s_squared))
    return u, s_squared, A, y_max
