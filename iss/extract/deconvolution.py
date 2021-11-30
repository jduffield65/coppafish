import numpy as np
from .. import utils
from flowdec import restoration as fd_restoration
from flowdec import data as fd_data
from flowdec import psf as fd_psf
from ..find_spots.base import detect_spots
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
import configparser
from ..setup.config import _option_formatters as config_format
import tifffile
import tensorflow as tf
import matplotlib


def get_isolated_points(spot_yx, isolation_dist):
    """
    get the isolated points in a point cloud as those whose neighbour is far.

    :param spot_yx: numpy integer array [n_peaks x image.ndim]
        yx or yxz location of spots found in image.
    :param isolation_dist: float
        spots are isolated if nearest neighbour is further away than this.
    :return: numpy boolean array [n_peaks]
    """
    nbrs = NearestNeighbors(n_neighbors=2).fit(spot_yx)
    distances, _ = nbrs.kneighbors(spot_yx)
    return distances[:, 1] > isolation_dist


def get_spot_images(image, spot_yx, shape):
    """
    builds an image around each spot of size given by shape and returns array containing all of these.

    :param image: numpy array [nY x nX (x nZ)]
        image spots were found on.
    :param spot_yx: numpy integer array [n_peaks x image.ndim]
        yx or yxz location of spots found.
    :param shape: list or numpy integer array giving size in y, x (and z) directions.
        desired size of image for each spot.
    :return: numpy array [n_peaks x y_shape x x_shape (x z_shape)]
    """
    if min(np.array(shape) % 2) == 0:
        raise ValueError(f"Require shape to be odd in each dimension but given shape was {shape}.")
    mid_index = np.ceil(np.array(shape)/2).astype(int) - 1  # index in spot_images where max intensity is for each spot.
    spot_images = np.empty((spot_yx.shape[0], *shape))
    spot_images[:] = np.nan  # set to nan if spot image goes out of bounds of image.
    max_image_index = np.array(image.shape)
    for s in tqdm(range(spot_yx.shape[0])):
        min_pos = np.clip((spot_yx[s] - mid_index), 0, max_image_index)
        max_pos = np.clip((spot_yx[s] + mid_index + 1), 0, max_image_index)
        spot_images_min_index = mid_index - (spot_yx[s] - min_pos)
        spot_images_max_index = mid_index + (max_pos - spot_yx[s])
        if len(shape) == 2:
            small_im = image[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1]]
            spot_images[s, spot_images_min_index[0]:spot_images_max_index[0],
                        spot_images_min_index[1]:spot_images_max_index[1]] = small_im
        elif len(shape) == 3:
            small_im = image[min_pos[0]:max_pos[0], min_pos[1]:max_pos[1], min_pos[2]:max_pos[2]]
            spot_images[s, spot_images_min_index[0]:spot_images_max_index[0],
                        spot_images_min_index[1]:spot_images_max_index[1],
                        spot_images_min_index[2]:spot_images_max_index[2]] = small_im
    return spot_images


def get_average_spot_image(spot_images, av_type='mean', symmetry=None, annulus_width=1.0):
    """
    given an array of spot images, this returns the average spot image.

    :param spot_images: numpy array [n_peaks x y_shape x x_shape (x z_shape)]
        array containing small images surrounding the n_peaks spots.
    :param av_type: 'mean' or 'median' indicating which average to use. optional.
        default: 'mean'
    :param symmetry: optional, default: None.
        None - just finds mean at every pixel.
        'quadrant_2d' - assumes each quadrant of each z-plane expected to look the same so concatenates these.
        'annulus' - assumes each z-plane is circularly symmetric about central pixel.
                    I.e. only finds only pixel value from all pixels a certain distance from centre.
    :param annulus_width: float, optional. default: 1.0
        if symmetry is 'annulus', this specifies how big an annulus to use, within which we expect all
        pixel values to be the same.
    :return: numpy array [y_shape x x_shape (x z_shape)]
    """
    if av_type == 'mean':
        av_func = lambda x, axis: np.nanmean(x, axis)
    elif av_type == 'median':
        av_func = lambda x, axis: np.nanmedian(x, axis)
    else:
        raise ValueError(f"av_type must be 'mean' or 'median' but value given was {av_type}")

    mid_index = np.ceil(np.array(spot_images.shape[1:]) / 2).astype(int) - 1

    if symmetry is None:
        av_image = av_func(spot_images, 0)
    elif symmetry == "quadrant_2d":
        # rotate all quadrants so spot is at bottom right corner
        quad1 = spot_images[:, 0:mid_index[0]+1, 0:mid_index[1]+1]
        quad2 = np.rot90(spot_images[:, 0:mid_index[0]+1, mid_index[1]:], 1, axes=(1, 2))
        quad3 = np.rot90(spot_images[:, mid_index[0]:, mid_index[1]:], 2, axes=(1, 2))
        quad4 = np.rot90(spot_images[:, mid_index[0]:, 0:mid_index[1]+1], 3, axes=(1, 2))
        all_quads = np.concatenate((quad1, quad2, quad3, quad4))
        av_quad = av_func(all_quads, 0)
        if spot_images.ndim == 4:
            av_image = np.pad(av_quad, [[0, mid_index[0]+1], [0, mid_index[1]+1], [0, 0]], 'symmetric')
        else:
            av_image = np.pad(av_quad, [[0, mid_index[0]+1], [0, mid_index[1]+1]], 'symmetric')
        # remove repeated central column and row
        av_image = np.delete(av_image, mid_index[0] + 1, axis=0)
        av_image = np.delete(av_image, mid_index[1] + 1, axis=1)
    elif symmetry == "annulus_2d":
        X, Y = np.meshgrid(np.arange(spot_images.shape[1]) - mid_index[0],
                           np.arange(spot_images.shape[2]) - mid_index[1])
        d = np.sqrt(X ** 2 + Y ** 2)
        annulus_bins = np.arange(0, d.max(), annulus_width)
        # find which bin each pixel should contribute to.
        bin_index = np.abs(np.expand_dims(d, 2) - annulus_bins).argmin(axis=2)
        av_image = np.zeros_like(spot_images[0])
        for i in range(annulus_bins.size):
            current_bin = bin_index == i
            av_image[current_bin] = av_func(spot_images[:, current_bin], (0, 1))
    else:
        raise ValueError(f"symmetry must be None, 'quadrant_2d' or 'annulus_2d' but value given was {symmetry}")
    return av_image


def psf_norm(x):
    """
    takes each image indicated by first axis and subtracts by image min and then divides by central pixel.
    So each image will now have a min value of 0 and a central value of 1.
    
    :param x: numpy array [n_images x im_sz_0 x im_sz_1 x ...]
    :return: numpy array [n_images x im_sz_0 x im_sz_1 x ...]
    """
    image_dims = tuple(np.arange(1, x.ndim))
    x_min = np.expand_dims(np.nanmin(x, image_dims), image_dims)
    x = x - x_min
    mid_index = np.ceil(np.array(x.shape[1:]) / 2).astype(int) - 1
    if x.ndim == 4:
        x_mid = np.expand_dims(x[:, mid_index[0], mid_index[1], mid_index[2]], image_dims)
    elif x.ndim == 3:
        x_mid = np.expand_dims(x[:, mid_index[0], mid_index[1]], image_dims)
    else:
        raise ValueError(f"Require x to be 3 or 4 dimensional but it has {x.ndim} dimensions.")
    return x / x_mid


def plot_psf(psf, n_columns=2, log=False):
    """
    plot psf as a series of panels for each z-plane.

    :param psf: numpy array [y_shape x x_shape (x z_shape)]
    :param n_columns: number of columns to have in subplots.
    :param log: whether to take log10 of psf before plotting
    """
    n_rows = np.ceil(psf.shape[2]/n_columns).astype(int)
    fig, axs = plt.subplots(n_rows, n_columns, sharex='all', sharey='all')
    fig.set_figheight(n_rows*3)
    fig.set_figwidth((n_columns+1)*3)
    z = 0
    if log:
        psf = np.log10(psf)
    caxis_min = np.percentile(psf, 1)
    caxis_max = psf.max()
    for i in range(n_columns):
        for j in range(n_rows):
            if z < psf.shape[2]:
                im = axs[j, i].imshow(psf[:, :, z], vmin=caxis_min, vmax=caxis_max)
                axs[j, i].set_title(f"z = {z}", fontsize=12)
                axs[j, i].xaxis.set_visible(False)
                axs[j, i].yaxis.set_visible(False)
                z += 1
            else:
               fig.delaxes(axs[j, i])
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist())
    plt.show()

# GibsonLanni Point Spread Function - symmetric, doesn't work.
# get psf info, need to do this for each channel
# c = 4  # anchor is channel 5 in MATLAB
# nd2_info = nd2.ND2File(nd2_file)
# channel_info = nd2_info.metadata.channels[c]
# psf_data = {"na": channel_info.microscope.objectiveNumericalAperture,
#             "wavelength": channel_info.channel.emissionLambdaNm/1000,
#             "size_z": nd2_info.sizes['Z'], "size_x": nd2_info.sizes['X'], "size_y": nd2_info.sizes['Y'],
#             "res_lateral": channel_info.volume.axesCalibration[0], "res_axial": channel_info.volume.axesCalibration[2],
#             "ni0": channel_info.microscope.immersionRefractiveIndex, "m": channel_info.microscope.objectiveMagnification}
# psf_file = output_dir + 'psf.json'
# with open(psf_file, 'w') as outfile:
#     json.dump(psf_data, outfile)
# psf = fd_psf.GibsonLanni.load(psf_file).generate()


if __name__ == '__main__':
    # get paramaters from config file
    config_file = "/Users/joshduffield/Documents/UCL/ISS/spin_disk/210805 seq w new disk 2/anchor_t10c5_python2.ini"
    config_file = "/Volumes/Volume/UCL/wf on thick section anchor +sst640/cf_deconvolve_params0.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    params = config['params']


    # nd2_file = '/Volumes/Subjects/ISS/Izzie/210805 seq w new disk 2/anchor.nd2'

    # load in image
    im_file = config_format['str'](params['input_dir']) + config_format['str'](params['input_image_name'])
    image = utils.tiff.load(im_file).astype(int)

    # get psf
    spot_yx, _ = detect_spots(image, intensity_thresh=config_format['number']((params['spot_intensity_thresh'])),
                              radius_xy=config_format['int'](params['spot_radius_xy']),
                              radius_z=config_format['int'](params['spot_radius_z']))
    isolated = get_isolated_points(spot_yx, isolation_dist=config_format['number'](params['isolation_dist']))
    spot_yx = spot_yx[isolated, :]
    spot_images = get_spot_images(image, spot_yx, shape=config_format['list_int'](params['spot_shape']))
    psf = get_average_spot_image(spot_images, config_format['str'](params['av_method']),
                                 config_format['str'](params['spot_symmetry']),
                                 annulus_width=config_format['number'](params['annulus_width']))
    # psf from GibsonLanni has min as 0 and 1 as max so normalise to match this.
    psf = psf - psf.min()
    psf = psf / psf.max()
    # matplotlib.use('TkAgg')  # so plot opens in pop out window
    plot_psf(psf, 3, True)

    # run deconvolution
    acq = fd_data.Acquisition(data=np.moveaxis(image, -1, 0), kernel=np.moveaxis(psf, -1, 0))
    algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_min=[1, 1, 1]).initialize()
    res = algo.run(acq, niter=config_format['int'](params['n_iter']))

    # # for running on GPU
    # acq = fd_data.Acquisition(data=np.moveaxis(image, -1, 0), kernel=np.moveaxis(psf, -1, 0))
    # algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_min=[1, 1, 1]).initialize()
    # # below uses less memory than above but sometimes get negative pad error
    # # algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_mode='none').initialize()
    # session_config = tf.compat.v1.ConfigProto()
    # session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # res = algo.run(acq, niter=config_format['int'](params['n_iter']), session_config=session_config)

    im_out = np.round(res.data).astype(np.uint16)
    im_out_file = config_format['str'](params['output_dir']) + config_format['str'](params['output_image_name'])
    tifffile.imwrite(im_out_file, im_out)
    hi = 5
