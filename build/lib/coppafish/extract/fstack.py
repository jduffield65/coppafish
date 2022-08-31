import numpy as np
from scipy import ndimage
from scipy.signal import medfilt2d
import cv2
from typing import Optional, Tuple


def rgb2gray(im: np.ndarray) -> np.ndarray:
    """
    Converts RGB ```m x n x 3``` color image into ```m x n``` greyscale image.
    Uses weighted sum indicated [here](https://www.mathworks.com/help/matlab/ref/rgb2gray.html).

    Args:
        im: ```float [m x n x 3]```. RGB image

    Returns:
        ```float [m x n]```. Greyscale image.

    """
    im_R = im[:, :, 0]
    im_G = im[:, :, 1]
    im_B = im[:, :, 2]
    if np.issubdtype(im.dtype,np.integer):
        im_output = np.round((0.2989 * im_R + 0.5870 * im_G + 0.1140 * im_B)).astype(im.dtype)
    else:
        im_output = (0.2989 * im_R + 0.5870 * im_G + 0.1140 * im_B).astype(im.dtype)
    return im_output


def im2double(im: np.ndarray) -> np.ndarray:
    """
    Equivalent to Matlab im2double function.
    Follows answer
    [here](https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python/29104511).

    Args:
        im: ```int```/```uint``````[m x n x ...]```. Image tom convert.

    Returns:
        ```float [m x n x ...]```. Converted image.
    """
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    return im.astype(float) / info.max  # Divide all values by the largest possible value in the datatype


def focus_stack(im_stack: np.ndarray, nhsize: int = 9, focus: Optional[np.ndarray] = None, alpha: float = 0.2,
                sth: float = 13) -> np.ndarray:
    """
    Generate extended depth-of-field image from focus sequence
    using noise-robust selective all-in-focus algorithm [1].
    Input images may be grayscale or color. For color images,
    the algorithm is applied to each color plane independently.

    For further details, see:
    [1] Pertuz et. al. "Generation of all-in-focus images by
      noise-robust selective fusion of limited depth-of-field
      images" IEEE Trans. Image Process, 22(3):1242 - 1251, 2013.

    S. Pertuz, Jan/2016

    Modified by Josh, 2021

    Args:
        im_stack: Element ```[:,:,p,:]``` is greyscale or RGB image at z-plane ```p```.
            RGB: ```uint8 [M x N x P x 3]```
            Gray: ```uint16 [M x N x P]```
        nhsize: Size of default window.
        focus: ```int [P]``` or ```None```.
            Vector with focus of each frame. If ```None```, will use ```np.arange(P)```.
        alpha: A scalar in ```[0,1]```. See [1] for details.
        sth: A scalar. See [1] for details.

    Returns:
        ```int [M x N]```. All In Focus (AIF) image.
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


def get_fmeasure(im_stack: np.ndarray, nhsize: int) -> np.ndarray:
    """
    Returns focus measure value for each pixel.

    Args:
        im_stack: Element ```[:,:,p,:]``` is greyscale or RGB image at z-plane ```p```.
            RGB: ```uint8 [M x N x P x 3]```.
            Gray: ```uint16 [M x N x P]```.
        nhsize: Size of focus measure window. Typical: ```M/200```.

    Returns:
        ```float [M x N x P]```. Focus measure image.
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


def get_smeasure(fm: np.ndarray, nhsize: int, focus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns selectivity measure value for each pixel.

    Args:
        fm: ```float [M x N x P]```.
            Focus Measure Image.
        nhsize: Size of focus measure window. Typical: ```M/200```.
        focus: ```int [P]```.
            Vector with focus of each frame. Typical: ```np.arange(P)```.

    Returns:
        - ```S``` - ```float [M x N]```. Selectivity measure image.
        - ```fm``` - ```float [M x N x P]```. Normalised focus measure image.
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


def get_weights(S: np.ndarray, fm: np.ndarray, alpha: float, sth: float) -> np.ndarray:
    """
    Computes sharpening parameter phi and then
    returns cut off frequency for high pass convolve_2d, ```omega```.

    Args:
        S: ```float [M x N]```.
            Selectivity measure image.
        fm: ```float [M x N x P]```.
            Normalised focus measure image.
        alpha: A scalar in ```[0, 1]```. Typical: ```0.2```.
        sth: A scalar. Typical: ```13```.

    Returns:
        ```float [M x N]```. Cut off frequency for high pass convolve_2d.
    """
    phi = 0.5 * (1 + np.tanh(alpha * (S - sth))) / alpha
    phi = medfilt2d(phi, 3)
    omega = 0.5 + 0.5 * np.tanh(np.expand_dims(phi, 2) * (fm - 1))
    return omega


def gfocus(im: np.ndarray, w_size: int) -> np.ndarray:
    """
    Compute focus measure using gray level local variance.

    Args:
        im: ```float [M x N]```.
            Gray scale image.
        w_size: Size of convolve_2d window. Typical: ```M/200```.

    Returns:
        ```float [M x N]```. Focus measure image.
    """
    mean_f = np.ones((w_size, w_size)) / (w_size ** 2)
    #u = ndimage.correlate(im, mean_f, mode='nearest')
    u = cv2.filter2D(im, -1, mean_f, borderType=cv2.BORDER_REPLICATE)
    fm = (im - u) ** 2
    fm = cv2.filter2D(fm, -1, mean_f, borderType=cv2.BORDER_REPLICATE)
    #fm = ndimage.correlate(fm, mean_f, mode='nearest')
    return fm


def gauss3p(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast 3-point gaussian interpolation.

    Args:
        x: ```int [P]```.
            Vector with focus of each frame. Typical: ```np.arange(P)```.
        y: ```float [M x N x P]```.
            Image to interpolate.

    Returns:
        - ```u``` - ```float [M x N]```.
            Mean value of gaussian function.
        - ```s_squared``` - ```float [M x N]```.
            Variance.
        - ```A``` - ```float [M x N]```.
            Max value of gaussian function.
        - ```y_max``` - ```float [M x N]```.
            Max projection of image ```y```.
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
