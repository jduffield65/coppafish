from math import floor
import cv2
import numpy as np
from .morphology import dilate


def periodic_line(p, v):
    """
    creates a flat structuring element
    containing 2*p+1 members.  v is a two-element vector containing
    integer-valued row and column offsets.  One structuring element member
    is located at the origin.  The other members are located at 1*v, -1*v,
    2*v, -2*v, ..., p*v, -p*v.
    copy of MATLAB strel('periodicline')
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


def disk(r, n=4):
    """
    creates a flat disk-shaped structuring element
    with the specified radius, r.  r must be a nonnegative integer.  n must
    be 0, 4, 6, or 8.  When n is greater than 0, the disk-shaped structuring
    element is approximated by a sequence of n (or sometimes n+2)
    periodic-line structuring elements.  When n is 0, no approximation is
    used, and the structuring element members comprise all pixels whose
    centers are no greater than r away from the origin.  n can be omitted,
    in which case its default value is 4.  Note: Morphological operations
    using disk approximations (n>0) run much faster than when n=0.  Also,
    the structuring elements resulting from choosing n>0 are suitable for
    computing granulometries, which is not the case for n=0.  Sometimes it
    is necessary for STREL to use two extra line structuring elements in the
    approximation, in which case the number of decomposed structuring
    elements used is n+2.
    copy of MATLAB strel('disk')
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


def disk_3d(r_xy, r_z):
    """
    gets structuring element used to find spots when dilated with 3d image.

    :param r_xy: integer
    :param r_z: integer
    :return: numpy integer array [2*r_xy+1, 2*r_xy+1, 2*r_z+1]. Each element either 0 or 1.
    """
    y, x, z = np.meshgrid(np.arange(-r_xy, r_xy + 1), np.arange(-r_xy, r_xy + 1), np.arange(-r_z, r_z + 1))
    se = x ** 2 + y ** 2 + z ** 2 <= r_xy ** 2
    return se.astype(int)


def annulus(r0, r_xy, r_z=None):
    """
    gets structuring element used to assess if spot isolated

    :param r0: float
        inner radius within which values are all zero.
    :param r_xy: float
        outer radius in xy direction.
        can be float not integer because all values with radius < r_xy1 and > r0 will be set to 1.
    :param r_z: float, optional
        outer radius in z direction. (size in z-pixels not normalised to xy pixel size).
        default: None meaning 2d annulus.
    :return: numpy integer array [2*floor(r_xy1)+1, 2*floor(r_xy1)+1, 2*floor(r_z1)+1]. Each element either 0 or 1.
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
