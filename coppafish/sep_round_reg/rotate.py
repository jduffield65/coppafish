import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import exposure
from skimage.transform import warp_polar, rotate
from skimage.filters import window
from skimage.registration import phase_cross_correlation

def process_image(A, z_planes,gamma,y,x,len):

    if z_planes != None:
        # First, collapse some z-planes
        img = np.zeros((A.shape[1], A.shape[2]))
        for i in z_planes:
            img += A[i]
        A = img

    A = A[y:y+len,x:x+len]

    # Next, make sure the contrast is well-adjusted
    A = exposure.equalize_hist(A)

    # Rescale so max = 1
    max = np.array(A).max()
    A = A / max

    # Invert the image
    A = np.ones(A.shape) - A

    A = A**gamma

    # Apply Hann Window to Images 1 and 2
    A = A * (window('hann', A.shape) ** 0.1)

    return A

def detect_rotation(ref, extra):

    # work with shifted FFT log-magnitudes
    ref_ft = np.log2(np.abs(fftshift(fft2(ref))))
    extra_ft = np.log2(np.abs(fftshift(fft2(extra))))

    # Plot image 1 and image 2 side by side
    plt.subplot(1, 2, 1)
    plt.imshow(ref_ft, cmap=plt.cm.gray)
    # plot 2:
    plt.subplot(1, 2, 2)
    plt.imshow(extra_ft, cmap=plt.cm.gray)
    plt.show()

    # Create log-polar transformed FFT mag images and register
    shape = ref_ft.shape
    radius = shape[0] // 4  # only take lower frequencies
    warped_ref_ft = warp_polar(ref_ft, radius=radius, scaling='log')
    warped_extra_ft = warp_polar(extra_ft, radius=radius, scaling='log')

    # Plot image 1 and image 2 side by side
    plt.subplot(1, 2, 1)
    plt.imshow(warped_ref_ft, cmap=plt.cm.gray)
    # plot 2:
    plt.subplot(1, 2, 2)
    plt.imshow(warped_extra_ft, cmap=plt.cm.gray)
    plt.show()

    warped_ref_ft = warped_ref_ft[:shape[0] // 2, :]  # only use half of FFT
    warped_extra_ft = warped_extra_ft[:shape[0] // 2, :]
    shifts, error, phasediff = phase_cross_correlation(warped_ref_ft,
                                                       warped_extra_ft,
                                                       upsample_factor=10,
                                                       normalization=None)

    # Use translation parameters to calculate rotation parameter
    shift_angle = shifts[0]

    return shift_angle, error

def patch_together(tile_dir,tilepos_yx,z_planes):

    """"This function simply creates a large stitched image from the reference round of a notebook by adding tiles
    """