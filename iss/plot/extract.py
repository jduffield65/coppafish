import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RangeSlider

import iss.utils.nd2
from ..pipeline.basic_info import set_basic_info
from .. import setup, extract, utils
import os


class plot_3d:
    def __init__(self, image, title, color_ax_min=None, color_ax_max=None):
        self.fig, self.ax = plt.subplots(1, 1)
        self.title = title
        self.nz = image.shape[2]
        self.z = 0
        self.base_image = image
        self.image = image.copy()
        # min_color is min possible color axis
        # color_ax_min is current min color axis.
        min_color = self.image.min()
        if color_ax_min is None:
            color_ax_min = min_color
        if color_ax_min < min_color:
            min_color = color_ax_min
        max_color = self.image.max()
        if color_ax_max is None:
            color_ax_max = self.image.max()
        if color_ax_max > max_color:
            max_color = color_ax_max
        self.im = self.ax.imshow(self.image[:, :, self.z], vmin=color_ax_min, vmax=color_ax_max)

        self.update()
        self.ax.invert_yaxis()
        self.fig.colorbar(self.im)
        self.fig.canvas.mpl_connect('scroll_event', self.z_scroll)

        slider_ax = self.fig.add_axes([0.1, 0.1, 0.01, 0.8])
        self.color_slider = RangeSlider(slider_ax, "Clim", min_color, max_color, [color_ax_min, color_ax_max],
                                        orientation='vertical', valfmt='%.2f')
        self.color_slider.on_changed(self.change_clim)

    def z_scroll(self, event):
        if event.button == 'up':
            self.z = (self.z + 1) % self.nz
        else:
            self.z = (self.z - 1) % self.nz
        self.update()

    def update(self):
        self.im.set_data(self.image[:, :, self.z])
        self.ax.set_title(f'{self.title}\nZ-Plane = {self.z}')
        self.im.axes.figure.canvas.draw()

    def change_clim(self, val):
        self.im.set_clim(val[0], val[1])
        self.im.axes.figure.canvas.draw()


class plot_filter:
    def __init__(self, config_file, t=None, r=None, c=None):
        print('Getting info from nd2 files...')
        config = setup.get_config(config_file)
        nbp_file, nbp_basic = set_basic_info(config['file_names'], config['basic_info'])
        print('Done.')

        # Get filter info
        config_extract = config['extract']
        r1 = config_extract['r1']
        r2 = config_extract['r2']
        if r1 is None:
            r1 = extract.get_pixel_length(config_extract['r1_auto_microns'], nbp_basic.pixel_size_xy)
        if r2 is None:
            r2 = r1 * 2
        self.filter_kernel = utils.morphology.hanning_diff(r1, r2)

        # Get deconvolution info - uses anchor round
        if os.path.isfile(nbp_file.psf):
            psf_tiles_used = [0]
            psf = utils.tiff.load(nbp_file.psf).astype(float)
        else:
            print('Getting psf...')
            if nbp_basic.ref_round == nbp_basic.anchor_round:
                im_file = os.path.join(nbp_file.input_dir, nbp_file.anchor + nbp_file.raw_extension)
            else:
                im_file = os.path.join(nbp_file.input_dir,
                                       nbp_file.round[nbp_basic.ref_round] + nbp_file.raw_extension)

            spot_images, psf_intensity_thresh, psf_tiles_used = \
                extract.get_psf_spots(im_file, nbp_basic.tilepos_yx, nbp_basic.tilepos_yx_nd2,
                                      nbp_basic.use_tiles, nbp_basic.ref_channel, nbp_basic.use_z,
                                      config_extract['psf_detect_radius_xy'], config_extract['psf_detect_radius_z'],
                                      config_extract['psf_min_spots'], config_extract['psf_intensity_thresh'],
                                      config_extract['auto_thresh_multiplier'], config_extract['psf_isolation_dist'],
                                      config_extract['psf_shape'])
            psf = extract.get_psf(spot_images, config_extract['psf_annulus_width'])
            print('Done.')
        # normalise psf so min is 0 and max is 1.
        psf = psf - psf.min()
        psf = psf / psf.max()
        self.wiener_pad_shape = config_extract['wiener_pad_shape']
        self.pad_im_shape = np.array([nbp_basic.tile_sz, nbp_basic.tile_sz, nbp_basic.nz]) + \
                       np.array(self.wiener_pad_shape) * 2
        self.psf = psf
        psf_plot = plot_3d(psf, 'Point Spread Function', 0, 1)
        self.wiener_constant = config_extract['wiener_constant']

        # Load in image to filter
        if t is None:
            t = psf_tiles_used[0]
        if r is None:
            r = nbp_basic.ref_round
        if c is None:
            c = nbp_basic.ref_channel
        self.title = f"Tile {t}, Round {r}, Channel {c}"
        if nbp_basic.use_anchor:
            # always have anchor as first round after imaging rounds
            round_files = nbp_file.round + [nbp_file.anchor]
        else:
            round_files = nbp_file.round
        print(f'Loading in tile {t}, round {r}, channel {c} from nd2 file...')
        im_file = os.path.join(nbp_file.input_dir, round_files[r] + nbp_file.raw_extension)
        images = utils.nd2.load(im_file)
        self.base_image = utils.nd2.get_image(images, iss.utils.nd2.get_nd2_tile_ind(t, nbp_basic.tilepos_yx_nd2,
                                                                                     nbp_basic.tilepos_yx),
                                              c, nbp_basic.use_z)
        print('Done.')

        # Plot image
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.subplots_adjust(left=0, right=1, bottom=0.16, top=0.94)
        self.nz = self.base_image.shape[2]
        self.z = 0
        self.wiener = False
        self.hanning_filter = False
        self.image = self.base_image.copy()
        self.im = self.ax.imshow(self.image[:, :, self.z], cmap="gray")
        self.raw_clims = [0, self.image.max()]
        self.filt_clims = [-nbp_basic.tile_pixel_value_shift, config_extract['scale_norm']]
        self.ax.invert_yaxis()
        self.fig.colorbar(self.im)
        self.fig.canvas.mpl_connect('scroll_event', self.z_scroll)

        slider_ax = self.fig.add_axes([0.1, 0.1, 0.01, 0.8])
        self.color_slider = RangeSlider(slider_ax, "Clim", -nbp_basic.tile_pixel_value_shift, np.iinfo(np.uint16).max,
                                        self.raw_clims, orientation='vertical', valfmt='%.0f')
        self.color_slider.on_changed(self.change_clim)

        self.update()
        self.update_images()

        # [left, bottom, width, height]
        wiener_button_ax = self.fig.add_axes([0.8, 0.025+0.05, 0.1, 0.04])
        self.wiener_button = Button(wiener_button_ax, 'WienerFilter', hovercolor='0.975')
        self.wiener_button.on_clicked(self.wiener_button_click)
        filter_button_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.filter_button = Button(filter_button_ax, 'HanningFilter', hovercolor='0.975')
        self.filter_button.on_clicked(self.filter_button_click)
        text_ax = self.fig.add_axes([0.5, 0.025, 0.2, 0.04])
        self.text_box = TextBox(text_ax, 'Wiener Constant', self.wiener_constant)
        self.text_box.on_submit(self.text_update)

        # plt.show(block=True) # this is needed when called from PythonConsole
        plt.show()

    def z_scroll(self, event):
        if event.button == 'up':
            self.z = (self.z + 1) % self.nz
        else:
            self.z = (self.z - 1) % self.nz
        self.update()

    def wiener_button_click(self, event):
        if self.wiener:
            self.wiener = False
            if self.hanning_filter:
                self.image = self.filtered_image
            else:
                self.image = self.base_image
        else:
            self.wiener = True
            if self.hanning_filter:
                self.image = self.filtered_wiener_image
            else:
                self.image = self.wiener_image
        self.update()

    def filter_button_click(self, event):
        if self.hanning_filter:
            self.hanning_filter = False
            if self.wiener:
                self.image = self.wiener_image
            else:
                self.image = self.base_image
        else:
            self.hanning_filter = True
            if self.wiener:
                self.image = self.filtered_wiener_image
            else:
                self.image = self.filtered_image
        self.update()

    def text_update(self, text):
        # If change wiener_constant value, all images are found again.
        self.wiener_constant = float(text)
        self.update_images()
        self.image = self.base_image
        self.wiener = False
        self.hanning_filter = False
        self.update()

    def change_clim(self, val):
        if self.hanning_filter:
            self.filt_clims = val
        else:
            self.raw_clims = val
        self.im.set_clim(val[0], val[1])
        self.im.axes.figure.canvas.draw()

    def update(self):
        self.im.set_data(self.image[:, :, self.z])
        if self.hanning_filter:
            self.im.set_clim(self.filt_clims[0], self.filt_clims[1])
            self.color_slider.set_val(self.filt_clims)
        else:
            self.im.set_clim(self.raw_clims[0], self.raw_clims[1])
            self.color_slider.set_val(self.raw_clims)
        self.ax.set_title(f'{self.title}\nZ-Plane = {self.z}, WienerFilter = {self.wiener},'
                          f' HanningFilter = {self.hanning_filter}')
        self.im.axes.figure.canvas.draw()

    def update_images(self):
        im, bad_columns = extract.strip_hack(self.base_image)
        print('Doing Wiener Filtering...')
        self.wiener_filter = extract.get_wiener_filter(self.psf, self.pad_im_shape, self.wiener_constant)
        self.wiener_image = extract.wiener_deconvolve(im, self.wiener_pad_shape, self.wiener_filter)
        print('Doing Filtering...')
        self.filtered_wiener_image = utils.morphology.convolve_2d(self.wiener_image, self.filter_kernel)
        scale = self.filt_clims[1] / self.filtered_wiener_image.max()
        self.filtered_wiener_image = self.filtered_wiener_image * scale
        print('Doing Filtering...')
        self.filtered_image = utils.morphology.convolve_2d(im, self.filter_kernel)
        scale = self.filt_clims[1] / self.filtered_image.max()
        self.filtered_image = self.filtered_image * scale
        self.wiener_image[:, bad_columns] = 0
        self.filtered_wiener_image[:, bad_columns] = 0
        self.filtered_image[:, bad_columns] = 0
        print('Done.')
