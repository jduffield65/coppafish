import nd2
import json
import numpy as np
import tifffile
from flowdec import restoration as fd_restoration
from flowdec import data as fd_data
from flowdec import psf as fd_psf

input_dir = '/Users/joshduffield/Documents/UCL/ISS/spin_disk/210805 seq w new disk 2/'
output_dir = input_dir
nd2_file = '/Volumes/Subjects/ISS/Izzie/210805 seq w new disk 2/anchor.nd2'
input_image_name = 'anchor_t10c5.tif'
output_image_name = 'anchor_t10c5_deconvolved.tif'

# get psf info, need to do this for each channel
c = 4  # anchor is channel 5 in MATLAB
nd2_info = nd2.ND2File(nd2_file)
channel_info = nd2_info.metadata.channels[c]
psf_data = {"na": channel_info.microscope.objectiveNumericalAperture,
            "wavelength": channel_info.channel.emissionLambdaNm/1000,
            "size_z": nd2_info.sizes['Z'], "size_x": nd2_info.sizes['X'], "size_y": nd2_info.sizes['Y'],
            "res_lateral": channel_info.volume.axesCalibration[0], "res_axial": channel_info.volume.axesCalibration[2],
            "ni0": channel_info.microscope.immersionRefractiveIndex, "m": channel_info.microscope.objectiveMagnification}
psf_file = output_dir + 'psf.json'
with open(psf_file, 'w') as outfile:
    json.dump(psf_data, outfile)
psf = fd_psf.GibsonLanni.load(psf_file).generate()

# load in image
im_file = input_dir + input_image_name
image = tifffile.imread(im_file, key=np.arange(nd2_info.sizes['Z'])).astype(int)
acq = fd_data.Acquisition(data=image, kernel=psf)

#
algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=acq.data.ndim, pad_min=[1, 1, 1]).initialize()
res = algo.run(acq, niter=4)
im_out = np.round(res.data).astype(np.uint16)
im_out_file = output_dir + output_image_name
tifffile.imwrite(im_out_file, im_out)
hi = 5
