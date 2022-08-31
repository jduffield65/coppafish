import napari
import numpy as np
from ... import extract, utils
from ...setup import Notebook
from ..raw import get_raw_images, number_to_list, add_basic_info_no_save
from typing import Optional, Union, List
from qtpy.QtCore import Qt
from PyQt5.QtWidgets import QSlider
import time
import matplotlib.pyplot as plt

plt.style.use('dark_background')


class view_filter:
    def __init__(self, nb: Optional[Notebook] = None, t: int = 0,
                 r: int = 0,
                 c: int = 0,
                 use_z: Optional[Union[int, List[int]]] = None, config_file: Optional[str] = None):
        """
        Function to view filtering of raw data in napari.
        There will be 2 scrollbars in 3D.
        One to change between *raw/difference_of_hanning/difference_of_hanning+smoothed* and one to change z-plane.

        There are also sliders to change the parameters for the filtering/smoothing.
        When the sliders are changed, the time taken for the new filtering/smoothing
        will be printed to the console.

        !!! note
            When `r_smooth` is set to `[1, 1, 1]`, no smoothing will be performed.
            When this is the case, changing the filtering radius using the slider will
            be quicker because it will only do filtering and not any smoothing.

        If `r == anchor_round` and `c == dapi_channel`, the filtering will be tophat filtering and no smoothing
        will be allowed. Otherwise, the filtering will be convolution with a difference of hanning kernel.

        The current difference of hanning kernel can be viewed by pressing the 'h' key.

        !!! warning "Requires access to `nb.file_names.input_dir`"

        Args:
            nb: *Notebook* for experiment. If no *Notebook* exists, pass `config_file` instead.
            t: npy (as opposed to nd2 fov) tile index to view.
                For an experiment where the tiles are arranged in a 4 x 3 (ny x nx) grid, tile indices are indicated as
                below:

                | 2  | 1  | 0  |

                | 5  | 4  | 3  |

                | 8  | 7  | 6  |

                | 11 | 10 | 9  |
            r: round to view
            c: Channel to view.
            use_z: Which z-planes to load in from raw data. If `None`, will use load all z-planes (except from first
                one if `config['basic_info']['ignore_first_z_plane'] == True`).
            config_file: path to config file for experiment.
        """
        if nb is None:
            nb = Notebook(config_file=config_file)
        add_basic_info_no_save(nb)  # deal with case where there is no notebook yet
        if use_z is None:
            use_z = nb.basic_info.use_z
        t, r, c, use_z = number_to_list([t, r, c, use_z])
        self.image_raw = get_raw_images(nb, t, r, c, use_z)[0, 0, 0]

        self.is_3d = nb.basic_info.is_3d
        if not self.is_3d:
            self.image_raw = extract.focus_stack(self.image_raw)
            self.image_plot = np.zeros((3, nb.basic_info.tile_sz, nb.basic_info.tile_sz), dtype=np.int32)
            self.image_plot[0] = self.image_raw
        else:
            self.image_plot = np.zeros((3, len(use_z), nb.basic_info.tile_sz, nb.basic_info.tile_sz), dtype=np.int32)
            self.image_plot[0] = np.moveaxis(self.image_raw, 2, 0)  # put z axis first for plotting
        self.raw_max = self.image_raw.max()

        # find faulty columns and update self.image_raw so don't have to run strip_hack each time we filter
        self.image_raw, self.bad_columns = extract.strip_hack(self.image_raw)

        # Get default filter info
        # label for each image in image_plot
        self.ax0_labels = ['Raw', 'Difference of Hanning', 'Difference of Hanning and Smoothed']
        if not self.is_3d:
            self.ax0_labels[0] = 'Raw (Focus Stacked)'
        config = nb.get_config()['extract']
        if r[0] == nb.basic_info.anchor_round and c[0] == nb.basic_info.dapi_channel:
            self.dapi = True
            if config['r_dapi'] is None:
                if config['r_dapi_auto_microns'] is not None:
                    config['r_dapi'] = extract.get_pixel_length(config['r_dapi_auto_microns'],
                                                                nb.basic_info.pixel_size_xy)
                else:
                    config['r_dapi'] = 48  # good starting value
            self.r_filter = config['r_dapi']
            self.image_plot = self.image_plot[:2]  # no smoothing if dapi
            self.ax0_labels = self.ax0_labels[:2]
            self.update_filter_image()
            r_filter_lims = [10, 70]
        else:
            self.dapi = False
            self.r_filter = config['r1']
            r2 = config['r2']
            if self.r_filter is None:
                self.r_filter = extract.get_pixel_length(config['r1_auto_microns'], nb.basic_info.pixel_size_xy)
            if r2 is None:
                r2 = self.r_filter * 2
            r_filter_lims = [2, 10]
            self.update_filter_image(r2)
            self.r_filter2 = r2

            # Get default smoothing info
            if config['r_smooth'] is None:
                # start with no smoothing. Quicker to change filter params as no need to update smoothing too.
                config['r_smooth'] = [1, 1, 1]
            if not nb.basic_info.is_3d:
                config['r_smooth'] = config['r_smooth'][:2]
            self.r_smooth = config['r_smooth']
            self.update_smooth_image()

        self.viewer = napari.Viewer()
        self.viewer.add_image(self.image_plot, name=f"Tile {t[0]}, Round {r[0]}, Channel{c[0]}")
        # Set min image contrast to 0 for better comparison between images
        self.viewer.layers[0].contrast_limits = [0, 0.9 * self.viewer.layers[0].contrast_limits[1]]
        self.ax0_ind = 0
        self.viewer.dims.set_point(0, self.ax0_ind)  # set filter type to raw initially
        self.viewer.dims.events.current_step.connect(self.filter_type_status)

        self.filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_slider.setRange(r_filter_lims[0], r_filter_lims[1])
        self.filter_slider.setValue(self.r_filter)
        # When dragging, status will show r_filter value
        self.filter_slider.valueChanged.connect(lambda x: self.show_filter_radius(x))
        # On release of slider, filtered / smoothed images updated
        self.filter_slider.sliderReleased.connect(self.filter_slider_func)
        if self.dapi:
            filter_slider_name = 'Tophat kernel radius'
        else:
            filter_slider_name = 'Difference of Hanning Radius'
        self.viewer.window.add_dock_widget(self.filter_slider, area="left", name=filter_slider_name)

        if not self.dapi:
            self.smooth_yx_slider = QSlider(Qt.Orientation.Horizontal)
            self.smooth_yx_slider.setRange(1, 5)  # gets very slow with large values
            self.smooth_yx_slider.setValue(self.r_smooth[0])
            # When dragging, status will show r_smooth value
            self.smooth_yx_slider.valueChanged.connect(lambda x: self.show_smooth_radius_yx(x))
            # On release of slider, smoothed image updated
            self.smooth_yx_slider.sliderReleased.connect(self.smooth_slider_func)
            smooth_title = "Smooth Radius"
            if self.is_3d:
                smooth_title = smooth_title + " YX"
            self.viewer.window.add_dock_widget(self.smooth_yx_slider, area="left", name=smooth_title)

        if self.is_3d and not self.dapi:
            self.smooth_z_slider = QSlider(Qt.Orientation.Horizontal)
            self.smooth_z_slider.setRange(1, 5)  # gets very slow with large values
            self.smooth_z_slider.setValue(self.r_smooth[2])
            # When dragging, status will show r_smooth value
            self.smooth_z_slider.valueChanged.connect(lambda x: self.show_smooth_radius_z(x))
            # On release of slider, smoothed image updated
            self.smooth_z_slider.sliderReleased.connect(self.smooth_slider_func)
            self.viewer.window.add_dock_widget(self.smooth_z_slider, area="left", name="Smooth Radius Z")

        if self.is_3d:
            self.viewer.dims.axis_labels = ['Filter Method', 'z', 'y', 'x']
        else:
            self.viewer.dims.axis_labels = ['Filter Method', 'y', 'x']

        self.key_call_functions()
        napari.run()

    def update_filter_image(self, r2: Optional[int] = None):
        if r2 is None:
            r2 = self.r_filter * 2  # default outer hanning filter is double inner radius
        self.r_filter2 = r2
        time_start = time.time()
        if self.dapi:
            print(f"Updating filtered image with r_dapi = {self.r_filter}...")
            filter_kernel = utils.strel.disk(self.r_filter)
            image_filter = utils.morphology.top_hat(self.image_raw, filter_kernel)
        else:
            print(f"Updating filtered image with r1 = {self.r_filter} and r2 = {r2}...")
            filter_kernel = utils.morphology.hanning_diff(self.r_filter, r2)
            image_filter = utils.morphology.convolve_2d(self.image_raw, filter_kernel)
        time_end = time.time()
        image_filter[:, self.bad_columns] = 0
        if not self.dapi:
            # set max value to be same as image_raw
            image_filter = np.rint(image_filter / image_filter.max() * self.raw_max)
        if self.is_3d:
            image_filter = np.moveaxis(image_filter, 2, 0)  # put z axis first for plotting
        self.image_plot[1] = image_filter
        print("Finished in {:.2f} seconds".format(time_end - time_start))

    def update_smooth_image(self):
        if np.max(self.r_smooth) > 1:
            print(f"Updating smoothed image with r_smooth = {self.r_smooth}...")
            image_filter = self.image_plot[1]
            if self.is_3d:
                image_filter = np.moveaxis(image_filter, 0, 2)  # put z axis to end for smoothing
            smooth_kernel = np.ones(tuple(np.array(self.r_smooth, dtype=int) * 2 - 1))
            smooth_kernel = smooth_kernel / np.sum(smooth_kernel)
            time_start = time.time()
            image_smooth = utils.morphology.imfilter(image_filter, smooth_kernel, oa=False)
            image_smooth[:, self.bad_columns] = 0
            time_end = time.time()
            # set max value to be same as image_raw
            image_smooth = np.rint(image_smooth / image_smooth.max() * self.raw_max)
            if self.is_3d:
                image_smooth = np.moveaxis(image_smooth, 2, 0)  # put z axis first for plotting
            self.image_plot[2] = image_smooth
            print("Finished in {:.2f} seconds".format(time_end - time_start))
        else:
            self.image_plot[2] = self.image_plot[1]  # if radius of smoothing are all 1, same as no smoothing

    def filter_type_status(self, event):
        if event.value[0] != self.ax0_ind:
            # Whenever filter type axis is changed, indicate in bottom left corner.
            self.ax0_ind = event.value[0]
            self.viewer.status = f'Filter Type: {self.ax0_labels[self.ax0_ind]}'

    def show_filter_radius(self, val):
        if self.dapi:
            self.viewer.status = f"Tophat kernel radius = {val}"
        else:
            self.viewer.status = f"Difference of Hanning radii: r1 = {val}, r2 = {val * 2}"

    def filter_slider_func(self):
        # TODO: Only filter current z-plane showing as will be much quicker
        if self.r_filter != self.filter_slider.value():
            self.r_filter = self.filter_slider.value()
            self.update_filter_image()
            if not self.dapi:
                self.update_smooth_image()
            self.viewer.layers[0].data = self.image_plot

    def show_smooth_radius_yx(self, val):
        r_smooth_update = self.r_smooth.copy()
        r_smooth_update[0] = val
        r_smooth_update[1] = val
        if np.max(r_smooth_update) == 1:
            self.viewer.status = f"Smoothing Radius = {r_smooth_update} (No smoothing)"
        else:
            self.viewer.status = f"Smoothing Radius = {r_smooth_update}"

    def show_smooth_radius_z(self, val):
        r_smooth_update = self.r_smooth.copy()
        r_smooth_update[2] = val
        if np.max(r_smooth_update) == 1:
            self.viewer.status = f"Smoothing Radius = {r_smooth_update} (No smoothing)"
        else:
            self.viewer.status = f"Smoothing Radius = {r_smooth_update}"

    def smooth_slider_func(self):
        r_smooth_update = self.r_smooth.copy()
        r_smooth_update[0] = self.smooth_yx_slider.value()
        r_smooth_update[1] = self.smooth_yx_slider.value()
        if self.is_3d:
            r_smooth_update[2] = self.smooth_z_slider.value()
        if self.r_smooth != r_smooth_update:
            self.r_smooth = r_smooth_update
            self.update_smooth_image()
            self.viewer.layers[0].data = self.image_plot

    def key_call_functions(self):
        @self.viewer.bind_key('h')
        def view_hanning_kernel(viewer):
            view_hanning(self.r_filter, self.r_filter2)


def view_hanning(r1: int, r2: int):
    """
    Views the 1D version of the difference of hanning kernel (before *utils/morphology/ftrans2* applied to make it 2D).

    Args:
        r1: Inner radius
        r2: Outer radius
    """
    h_outer = np.hanning(2 * r2 + 3)[1:-1]  # ignore zero values at first and last index
    h_outer = -h_outer / h_outer.sum()
    h_inner = np.hanning(2 * r1 + 3)[1:-1]
    h_inner = h_inner / h_inner.sum()
    h = h_outer.copy()
    h[r2 - r1:r2 + r1 + 1] += h_inner
    h_inner_plot = np.zeros_like(h)
    h_inner_plot[r2 - r1:r2 + r1 + 1] += h_inner

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(-r2, r2+1)
    ax1.plot(x, h_outer, label='Negative Outer Hanning Window')
    ax1.plot(x, h_inner_plot, label='Positive Inner Hanning Window')
    ax1.plot(x, h, label='Difference of Hanning Kernel')
    ax1.legend()
    plt.title(f'1D view of Difference of Hanning Kernel with r1={r1}, r2={r2}')
    plt.show()
