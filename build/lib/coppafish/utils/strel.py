from math import floor, ceil
import cv2
import numpy as np
from .morphology import dilate
from typing import Optional


def periodic_line(p: int, v: np.ndarray) -> np.ndarray:
    """
    Creates a flat structuring element containing `2*p+1` members.

    `v` is a two-element vector containing integer-valued row and column offsets.

    One structuring element member is located at the origin.
    The other members are located at `1*v, -1*v, 2*v, -2*v, ..., p*v, -p*v`.

    Copy of MATLAB `strel('periodicline')`.
    """
    pp = np.repeat(np.arange(-p, p + 1).reshape(-1, 1), 2, axis=1)
    rc = pp * v
    r = rc[:, 0]
    c = rc[:, 1]
    M = 2 * np.abs(r).max() + 1
    N = 2 * np.abs(c).max() + 1
    nhood = np.zeros((M, N), dtype=bool)
    # idx = np.ravel_multi_index([r + np.abs(r).max(), c + np.abs(c).max()], (M, N))
    nhood[r + np.abs(r).max(), c + np.abs(c).max()] = True
    return nhood.astype(np.uint8)


def disk(r: int, n: int = 4) -> np.ndarray:
    """
    Creates a flat disk-shaped structuring element with the specified radius, `r`.

    `r` must be a nonnegative integer.

    `n` must be `0, 4, 6, or 8`.
    When `n` is greater than `0`, the disk-shaped structuring
    element is approximated by a sequence of `n` (or sometimes `n+2`)
    periodic-line structuring elements.
    When `n` is `0`, no approximation is used, and the structuring element members comprise all pixels whose
    centers are no greater than `r` away from the origin.  `n` can be omitted, in which case its default value is `4`.

    !!! note
        Morphological operations using disk approximations (`n>0`) run much faster than when `n=0`.
        Also, the structuring elements resulting from choosing `n>0` are suitable for
        computing granulometries, which is not the case for `vn=0`.  Sometimes it
        is necessary for STREL to use two extra line structuring elements in the
        approximation, in which case the number of decomposed structuring
        elements used is `n+2`.

    Copy of MATLAB `strel('disk')`.
    """
    if r < 3:
        # Radius is too small to use decomposition, so force n=0.
        n = 0
    if n == 0:
        xx, yy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        nhood = xx ** 2 + yy ** 2 <= r ** 2
    else:
        # Reference for radial decomposition of disks:  Rolf Adams, "Radial
        # Decomposition of Discs and Spheres," CVGIP:  Graphical Models and
        # Image Processing, vol. 55, no. 5, September 1993, pp. 325-332.
        #
        # The specific decomposition technique used here is radial
        # decomposition using periodic lines.  The reference is:  Ronald
        # Jones and Pierre Soille, "Periodic lines: Definition, cascades, and
        # application to granulometries," Pattern Recognition Letters,
        # vol. 17, 1996, pp. 1057-1063.
        #
        # Determine the set of "basis" vectors to be used for the
        # decomposition.  The rows of v will be used as offset vectors for
        # periodic line strels.
        if n == 4:
            v = np.array([[1, 0], [1, 1], [0, 1], [-1, 1]])
        elif n == 6:
            v = np.array([[1, 0], [1, 2], [2, 1], [0, 1], [-1, 2], [-2, 1]])
        elif n == 8:
            v = np.array([[1, 0], [2, 1], [1, 1], [1, 2], [0, 1], [-1, 2], [-1, 1], [-2, 1]])
        else:
            raise ValueError(f'Value of n provided ({n}) is not 0, 4, 6 or 8.')
        # Determine k, which is the desired radial extent of the periodic
        # line strels.  For the origin of this formula, see the second
        # paragraph on page 328 of the Rolf Adams paper.
        theta = np.pi / (2 * n)
        k = 2 * r / (1 / np.tan(theta) + 1 / np.sin(theta))

        # For each periodic line strel, determine the repetition parameter,
        # rp.  The use of floor() in the computation means that the resulting
        # strel will be a little small, but we will compensate for this
        # below.
        nhood = np.ones((2 * r - 1, 2 * r - 1), np.uint8) * -np.inf
        nhood[int((nhood.shape[0] - 1) / 2), int((nhood.shape[0] - 1) / 2)] = 1
        for q in range(n):
            rp = int(np.floor(k / np.linalg.norm(v[q, :])))
            decomposition = periodic_line(rp, v[q, :])
            nhood = dilate(nhood, decomposition)
        nhood = nhood > 0

        # Now we are going to add additional vertical and horizontal line
        # strels to compensate for the fact that the strel resulting from the
        # above decomposition tends to be smaller than the desired size.
        extra_strel_size = int(sum(np.sum(nhood, axis=1) == 0) + 1)
        if extra_strel_size > 0:
            # Update the computed neighborhood to reflect the additional strels in
            # the decomposition.
            nhood = cv2.dilate(nhood.astype(np.uint8), np.ones((1, extra_strel_size), dtype=np.uint8))
            nhood = cv2.dilate(nhood, np.ones((extra_strel_size, 1), dtype=np.uint8))
            nhood = nhood > 0
    return nhood.astype(int)


def disk_3d(r_xy: float, r_z: float) -> np.ndarray:
    """
    Gets structuring element used to find spots when dilated with 3d image.

    Args:
        r_xy: Radius in xy direction.
        r_z: Radius in z direction.

    Returns:
        `int [2*r_xy+1, 2*r_xy+1, 2*r_z+1]`.
            Structuring element with each element either `0` or `1`.
    """
    y, x, z = np.meshgrid(np.arange(-np.ceil(r_xy), np.ceil(r_xy) + 1), np.arange(-np.ceil(r_xy), np.ceil(r_xy) + 1),
                          np.arange(-np.ceil(r_z), np.ceil(r_z) + 1))
    se = x ** 2 + y ** 2 + z ** 2 <= r_xy ** 2
    # Crop se to remove zeros at extremities
    se = se[:, :, ~np.all(se == 0, axis=(0, 1))]
    se = se[:, ~np.all(se == 0, axis=(0, 2)), :]
    se = se[~np.all(se == 0, axis=(1, 2)), :, :]
    return se.astype(int)


def annulus(r0: float, r_xy: float, r_z: Optional[float] = None) -> np.ndarray:
    """
    Gets structuring element used to assess if spot isolated.

    Args:
        r0: Inner radius within which values are all zero.
        r_xy: Outer radius in xy direction.
            Can be float not integer because all values with `radius < r_xy1` and `> r0` will be set to `1`.
        r_z: Outer radius in z direction. Size in z-pixels.
            None means 2D annulus returned.

    Returns:
        `int [2*floor(r_xy1)+1, 2*floor(r_xy1)+1, 2*floor(r_z1)+1]`.
            Structuring element with each element either `0` or `1`.
    """
    r_xy1_int = floor(r_xy)
    if r_z is None:
        y, x = np.meshgrid(np.arange(-r_xy1_int, r_xy1_int + 1), np.arange(-r_xy1_int, r_xy1_int + 1))
        m = x ** 2 + y ** 2
    else:
        r_z1_int = floor(r_z)
        y, x, z = np.meshgrid(np.arange(-r_xy1_int, r_xy1_int + 1), np.arange(-r_xy1_int, r_xy1_int + 1),
                              np.arange(-r_z1_int, r_z1_int + 1))
        m = x ** 2 + y ** 2 + z ** 2
    # only use upper radius in xy direction as z direction has different pixel size.
    annulus = r_xy ** 2 >= m
    annulus = np.logical_and(annulus, m > r0 ** 2)
    return annulus.astype(int)


def fspecial(r_y: float, r_x: Optional[float] = None, r_z: Optional[float] = None) -> np.ndarray:
    """
    Creates an ellipsoidal 3D filter kernel if `r_y`, `r_x` and `r_z` given.
    Copy of MATlAB `fspecial3('ellipsoid')`.

    Creates a disk 2D filter kernel if just `r_y` given. Copy of MATlAB `fspecial('disk')`.

    Args:
        r_y: Radius in y direction or radius of disk if only parameter provided.
        r_x: Radius in x direction.
        r_z: Radius in z direction.

    Returns:
        `float [2*ceil(r_y)+1, 2*ceil(r_x)+1, 2*ceil(r_z)+1]`.
            Filtering kernel.
    """
    if r_x is None and r_z is None:
        r = r_y
        crad = ceil(r - 0.5)
        x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
        max_xy = np.maximum(np.abs(x), np.abs(y))
        min_xy = np.minimum(np.abs(x), np.abs(y))
        m1 = (r ** 2 < (max_xy + 0.5) ** 2 + (min_xy - 0.5) ** 2) * (min_xy - 0.5) + \
             (r ** 2 >= (max_xy + 0.5) ** 2 + (min_xy - 0.5) ** 2) * np.sqrt(r ** 2 - (max_xy + 0.5) ** 2,
                                                                             dtype=np.complex_)
        m1 = np.real(m1)
        m2 = (r ** 2 > (max_xy - 0.5) ** 2 + (min_xy + 0.5) ** 2) * (min_xy + 0.5) + \
             (r ** 2 <= (max_xy - 0.5) ** 2 + (min_xy + 0.5) ** 2) * np.sqrt(r ** 2 - (max_xy - 0.5) ** 2,
                                                                             dtype=np.complex_)
        m2 = np.real(m2)
        sgrid = (r ** 2 * (0.5 * (np.arcsin(m2 / r) - np.arcsin(m1 / r)) +
                           0.25 * (np.sin(2 * np.arcsin(m2 / r)) - np.sin(2 * np.arcsin(m1 / r)))) - (max_xy - 0.5) * (
                             m2 - m1) +
                 (m1 - min_xy + 0.5)) * ((((r ** 2 < (max_xy + 0.5) ** 2 + (min_xy + 0.5) ** 2) &
                                           (r ** 2 > (max_xy - 0.5) ** 2 + (min_xy - 0.5) ** 2)) |
                                          ((min_xy == 0) & (max_xy - 0.5 < r) & (max_xy + 0.5 >= r))))
        sgrid = sgrid + ((max_xy + 0.5) ** 2 + (min_xy + 0.5) ** 2 < r ** 2)
        sgrid[crad, crad] = min(np.pi * r ** 2, np.pi / 2)
        if crad > 0.0 and r > crad - 0.5 and r ** 2 < (crad - 0.5) ** 2 + 0.25:
            m1 = np.sqrt(r ** 2 - (crad - 0.5) ** 2)
            m1n = m1 / r
            sg0 = 2 * (r ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
            sgrid[2 * crad, crad] = sg0
            sgrid[crad, 2 * crad] = sg0
            sgrid[crad, 0] = sg0
            sgrid[0, crad] = sg0
            sgrid[2 * crad - 1, crad] = sgrid[2 * crad - 1, crad] - sg0
            sgrid[crad, 2 * crad - 1] = sgrid[crad, 2 * crad - 1] - sg0
            sgrid[crad, 1] = sgrid[crad, 1] - sg0
            sgrid[1, crad] = sgrid[1, crad + 1] - sg0
        sgrid[crad, crad] = min(sgrid[crad, crad], 1)
        h = sgrid / np.sum(sgrid)
    else:
        x, y, z = np.meshgrid(np.arange(-ceil(r_x), ceil(r_x) + 1), np.arange(-ceil(r_y), ceil(r_y) + 1),
                              np.arange(-ceil(r_z), ceil(r_z) + 1))
        h = (1 - x ** 2 / r_x ** 2 - y ** 2 / r_y ** 2 - z ** 2 / r_z ** 2) >= 0
        h = h / np.sum(h)
    return h
