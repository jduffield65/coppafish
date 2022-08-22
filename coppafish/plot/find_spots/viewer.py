import napari
import numpy as np
import os
import warnings
from typing import Optional
from ...setup import Notebook
from ..raw import get_raw_images, number_to_list, add_basic_info_no_save
from ... import extract, utils
from ...find_spots import check_neighbour_intensity
from qtpy.QtCore import Qt
from PyQt5.QtWidgets import QSlider
import time


def get_filtered_image(nb: Notebook, t: int, r: int, c: int) -> np.ndarray:
    # Function to load in raw image then filter according to parameters in `config['extract']`.

    # Load in raw image
    use_z = nb.basic_info.use_z
    t, r, c, use_z = number_to_list([t, r, c, use_z])
    image_raw = get_raw_images(nb, t, r, c, use_z)[0, 0, 0]

    # Filter image
    if not nb.basic_info.is_3d:
        image_raw = extract.focus_stack(image_raw)
    image_raw, bad_columns = extract.strip_hack(image_raw)
    config = nb.get_config()['extract']
    r1 = config['r1']
    r2 = config['r2']
    if r1 is None:
        r1 = extract.get_pixel_length(config['r1_auto_microns'], nb.basic_info.pixel_size_xy)
    if r2 is None:
        r2 = r1 * 2
    filter_kernel = utils.morphology.hanning_diff(r1, r2)
    image = utils.morphology.convolve_2d(image_raw, filter_kernel)

    # Smooth image
    if config['r_smooth'] is not None:
        smooth_kernel = np.ones(tuple(np.array(config['r_smooth'], dtype=int) * 2 - 1))
        smooth_kernel = smooth_kernel / np.sum(smooth_kernel)
        image = utils.morphology.imfilter(image, smooth_kernel, oa=False)
    image[:, bad_columns] = 0

    # Scale image
    scale = config['scale_norm'] / image.max()
    image = np.rint(image * scale).astype(np.int32)
    return image


class view_find_spots:
    def __init__(self, nb: Optional[Notebook] = None, t: int = 0, r: int = 0, c: int = 0,
                 show_isolated: bool = False, config_file: Optional[str] = None):
        """
        This viewer shows how spots are detected in an image.
        There are sliders to vary the parameters used for spot detection so the effect of them can be seen.

        Can also view points from `±z_thick` z-planes on current z-plane using the z thickness slider. Initially,
        z thickness will be 1.

        !!! warning "Requires access to `nb.file_names.input_dir` or `nb.file_names.tile_dir`"

        Args:
            nb: *Notebook* for experiment. If no *Notebook* exists, pass `config_file` instead. In this case,
                the raw images will be loaded and then filtered according to parameters in `config['extract']`.
            t: npy (as opposed to nd2 fov) tile index to view.
                For an experiment where the tiles are arranged in a 4 x 3 (ny x nx) grid, tile indices are indicated as
                below:

                | 2  | 1  | 0  |

                | 5  | 4  | 3  |

                | 8  | 7  | 6  |

                | 11 | 10 | 9  |
            r: round to view
            c: Channel to view.
            show_isolated: Spots which are identified as *isolated* in the anchor round/channel are used to compute
                the `bleed_matrix`. Can see which spots are isolated by setting this to `True`.
                Note, this is very slow in *3D*, around 300s for a 2048 x 2048 x 50 image.
            config_file: path to config file for experiment.
        """
        if nb is None:
            nb = Notebook(config_file=config_file)
        add_basic_info_no_save(nb)  # deal with case where there is no notebook yet

        if r == nb.basic_info.anchor_round:
            if c != nb.basic_info.anchor_channel:
                raise ValueError(f'No spots are found on round {r}, channel {c} in the pipeline.\n'
                                 f'Only spots on anchor_channel = {nb.basic_info.anchor_channel} used for the '
                                 f'anchor round.')
        if r == nb.basic_info.ref_round and c == nb.basic_info.ref_channel:
            if show_isolated:
                self.show_isolated = True
            else:
                self.show_isolated = False
        else:
            if show_isolated:
                warnings.warn(f'Not showing isolated spots as slow and isolated status not used for round {r},'
                              f' channel {c}')
            self.show_isolated = False

        self.is_3d = nb.basic_info.is_3d
        if self.is_3d:
            tile_file = nb.file_names.tile[t][r][c]
        else:
            tile_file = nb.file_names.tile[t][r]
        if not os.path.isfile(tile_file):
            warnings.warn(f"The file {tile_file}\ndoes not exist so loading raw image and filtering it")
            self.image = get_filtered_image(nb, t, r, c)
        else:
            self.image = utils.npy.load_tile(nb.file_names, nb.basic_info, t, r, c)
            scale = 1  # Can be any value as not actually used but needed as argument in get_extract_info

        # Get auto_threshold value used to detect spots
        if nb.has_page('extract'):
            self.auto_thresh = nb.extract.auto_thresh[t, r, c]
        else:
            config = nb.get_config()['extract']
            z_info = int(np.floor(nb.basic_info.nz / 2))
            hist_values = np.arange(-nb.basic_info.tile_pixel_value_shift,
                                    np.iinfo(np.uint16).max - nb.basic_info.tile_pixel_value_shift + 2, 1)
            hist_bin_edges = np.concatenate((hist_values - 0.5, hist_values[-1:] + 0.5))
            max_npy_pixel_value = np.iinfo(np.uint16).max - nb.basic_info.tile_pixel_value_shift
            self.auto_thresh = extract.get_extract_info(self.image, config['auto_thresh_multiplier'], hist_bin_edges,
                                                        max_npy_pixel_value, scale, z_info)[0]

        config = nb.get_config()['find_spots']
        self.r_xy = config['radius_xy']
        if self.is_3d:
            self.r_z = config['radius_z']
        else:
            self.r_z = None
        if config['isolation_thresh'] is None:
            config['isolation_thresh'] = self.auto_thresh * config['auto_isolation_thresh_multiplier']
        self.isolation_thresh = config['isolation_thresh']
        self.r_isolation_inner = config['isolation_radius_inner']
        self.r_isolation_xy = config['isolation_radius_xy']
        self.normal_color = np.array([1, 0, 0, 1]) # red
        self.isolation_color = np.array([0, 1, 0, 1])  # green
        self.neg_neighb_color = np.array([0, 0, 1, 1])  # blue
        self.point_size = 9
        self.z_thick = 1  # show +/- 1 plane initially
        self.z_thick_list = np.arange(1, 1 + 15 * 2, 2)  # only odd z-thick make any difference
        if self.is_3d:
            self.r_isolation_z = config['isolation_radius_z']
        else:
            self.r_isolation_z = None

        self.small = 1e-6  # for computing local maxima: shouldn't matter what it is (keep below 0.01 for int image).
        # perturb image by small amount so two neighbouring pixels that did have the same value now differ slightly.
        # hence when find maxima, will only get one of the pixels not both.
        rng = np.random.default_rng(0)   # So shift is always the same.
        # rand_shift must be larger than small to detect a single spot.
        rand_im_shift = rng.uniform(low=self.small*2, high=0.2, size=self.image.shape)
        self.image = self.image + rand_im_shift

        self.dilate = None
        self.spot_zyx = None
        self.image_isolated = None
        self.no_negative_neighbour = None
        self.update_dilate()
        if self.show_isolated:
            self.update_isolated_image()
        self.viewer = napari.Viewer()
        name = f"Tile {t}, Round {r}, Channel {c}"
        if self.is_3d:
            self.viewer.add_image(np.moveaxis(np.rint(self.image).astype(np.int32), 2, 0), name=name)
        else:
            self.viewer.add_image(np.rint(self.image).astype(np.int32), name=name)

        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 3 * self.auto_thresh)
        self.thresh_slider.setValue(self.auto_thresh)
        # When dragging, status will show auto_thresh value
        self.thresh_slider.valueChanged.connect(lambda x: self.show_thresh(x))
        # On release of slider, filtered / smoothed images updated
        self.thresh_slider.sliderReleased.connect(self.update_spots)
        self.viewer.window.add_dock_widget(self.thresh_slider, area="left", name='Intensity Threshold')
        if self.show_isolated:
            self.isolation_thresh_slider = QSlider(Qt.Orientation.Horizontal)
            self.isolation_thresh_slider.setRange(-2 * np.abs(self.isolation_thresh), 0)
            self.isolation_thresh_slider.setValue(self.isolation_thresh)
            # When dragging, status will show auto_thresh value
            self.isolation_thresh_slider.valueChanged.connect(lambda x: self.show_isolation_thresh(x))
            # On release of slider, filtered / smoothed images updated
            self.isolation_thresh_slider.sliderReleased.connect(self.update_isolated_spots)
        self.update_spots()

        self.r_xy_slider = QSlider(Qt.Orientation.Horizontal)
        self.r_xy_slider.setRange(2, 10)
        self.r_xy_slider.setValue(self.r_xy)
        # When dragging, status will show r_xy value
        self.r_xy_slider.valueChanged.connect(lambda x: self.show_radius_xy(x))
        # On release of slider, filtered / smoothed images updated
        self.r_xy_slider.sliderReleased.connect(self.radius_slider_func)
        self.viewer.window.add_dock_widget(self.r_xy_slider, area="left", name='Detection Radius YX')

        if self.is_3d:
            self.r_z_slider = QSlider(Qt.Orientation.Horizontal)
            self.r_z_slider.setRange(2, 12)
            self.r_z_slider.setValue(self.r_xy)
            # When dragging, status will show r_z value
            self.r_z_slider.valueChanged.connect(lambda x: self.show_radius_z(x))
            # On release of slider, filtered / smoothed images updated
            self.r_z_slider.sliderReleased.connect(self.radius_slider_func)
            self.viewer.window.add_dock_widget(self.r_z_slider, area="left", name='Detection Radius Z')

            self.z_thick_slider = QSlider(Qt.Orientation.Horizontal)
            self.z_thick_slider.setRange(0, int((self.z_thick_list[-1] - 1) / 2))
            self.z_thick_slider.setValue(self.z_thick)
            # When dragging, status will show r_z value
            self.z_thick_slider.valueChanged.connect(lambda x: self.change_z_thick(x))
            self.viewer.window.add_dock_widget(self.z_thick_slider, area="left", name='Z Thickness')

        if self.show_isolated:
            self.viewer.window.add_dock_widget(self.isolation_thresh_slider, area="left", name='Isolation Threshold')
        # set image as selected layer so can see intensity values in status
        self.viewer.layers.selection.active = self.viewer.layers[0]
        napari.run()

    def show_thresh(self, thresh):
        self.viewer.status = f"Intensity Threshold = {thresh}"

    def show_radius_xy(self, r_xy):
        self.viewer.status = f"Detection Radius YX = {r_xy}"

    def show_radius_z(self, r_z):
        self.viewer.status = f"Detection Radius Z = {r_z}"

    def update_dilate(self):
        # When radius changes, need to recompute dilation
        if self.r_z is not None:
            se = np.ones((2*self.r_xy-1, 2*self.r_xy-1, 2*self.r_z-1), dtype=int)
            print(f"Updating dilated image with r_xy = {self.r_xy}, r_z = {self.r_z}...")
        else:
            se = np.ones((2*self.r_xy-1, 2*self.r_xy-1), dtype=int)
            print(f"Updating dilated image with r_xy = {self.r_xy}...")
        time_start = time.time()
        self.dilate = utils.morphology.dilate(self.image, se)
        time_end = time.time()
        print("Finished in {:.2f} seconds".format(time_end - time_start))

    def update_spots(self):
        # When auto_thresh changes, don't need to recompute the dilation, just update new position of spots
        self.auto_thresh = self.thresh_slider.value()
        spots = np.logical_and(self.image + self.small > self.dilate, self.image > self.auto_thresh)
        peak_pos = np.where(spots)
        self.spot_zyx = np.concatenate([coord.reshape(-1, 1) for coord in peak_pos], axis=1)
        self.no_negative_neighbour = check_neighbour_intensity(self.image, self.spot_zyx,
                                                               thresh=0)
        if self.is_3d:
            self.spot_zyx = self.spot_zyx[:, [2, 0, 1]]
        if len(self.viewer.layers) == 1:
            if self.is_3d:
                point_size = [self.z_thick_list[self.z_thick], self.point_size, self.point_size]
            else:
                point_size = self.point_size
            self.viewer.add_points(self.spot_zyx, edge_color=self.normal_color, face_color=self.normal_color,
                                   symbol='x', opacity=0.8, edge_width=0, out_of_slice_display=True,
                                   size=point_size, name='Spots Found')
        else:
            self.viewer.layers[1].data = self.spot_zyx
        self.viewer.layers[1].face_color[self.no_negative_neighbour] = self.normal_color
        self.viewer.layers[1].face_color[np.invert(self.no_negative_neighbour)] = self.neg_neighb_color
        self.viewer.layers[1].visible = 1  # no idea why, but seem to need this line to update colors
        if self.show_isolated:
            self.update_isolated_spots()
        self.viewer.layers[1].selected_data = set()  # sometimes it selects points at random when change thresh

    def radius_slider_func(self):
        # When r_xy or r_z changed, this will recompute dilation and add new local maxima to plot.
        radius_changed = self.r_xy != self.r_xy_slider.value()
        if self.is_3d:
            radius_changed = np.any([radius_changed, self.r_z != self.r_z_slider.value()])
        if radius_changed:
            self.r_xy = self.r_xy_slider.value()
            if self.is_3d:
                self.r_z = self.r_z_slider.value()
            self.update_dilate()
            self.update_spots()

    def show_isolation_thresh(self, thresh):
        self.viewer.status = f"Isolation Threshold = {thresh}"

    def update_isolated_image(self):
        # Perfoms convolution of image with isolation annulus kernel.
        # Once done, don't need to do again if only changing isolation_thresh.
        # isolated image calculation is very slow, maybe make manual version without jax
        if self.is_3d:
            print(f"Updating isolation image with r_isolation_inner = {self.r_isolation_inner}, "
                  f"r_isolation_xy = {self.r_isolation_xy}, r_isolation_z = {self.r_isolation_z}...")
        else:
            print(f"Updating isolation image with r_isolation_inner = {self.r_isolation_inner}, "
                  f"r_isolation_xy = {self.r_isolation_xy}...")
        kernel = utils.strel.annulus(self.r_isolation_inner, self.r_isolation_xy, self.r_isolation_z)
        time_start = time.time()
        self.image_isolated = utils.morphology.imfilter(self.image, kernel, 0, 'corr', oa=False) / np.sum(kernel)
        time_end = time.time()
        print("Finished in {:.2f} seconds".format(time_end - time_start))
        if self.is_3d:
            self.image_isolated = np.moveaxis(self.image_isolated, 2, 0)  # put z first for reading off at spot_zyx.

    def update_isolated_spots(self):
        # when isolation_thresh changes, can show new spots without recomputing image_isolated.
        self.isolation_thresh = self.isolation_thresh_slider.value()
        isolated = self.image_isolated[tuple([self.spot_zyx[:, j] for j in
                                              range(self.image_isolated.ndim)])] < self.isolation_thresh
        self.viewer.layers[1].face_color[np.invert(isolated)] = self.normal_color
        isolated = np.logical_and(isolated, self.no_negative_neighbour)
        self.viewer.layers[1].face_color[np.invert(self.no_negative_neighbour)] = self.neg_neighb_color
        self.viewer.layers[1].face_color[isolated] = self.isolation_color
        self.viewer.layers[1].visible = 1  # no idea why, but seem to need this line to update colors

    def change_z_thick(self, z_thick):
        # Show spots from different z-planes
        # Only makes a difference when size is 1, 3, 5, 7 so make sure it is odd with z_thick_list
        self.viewer.status = f"Z-thickness = {z_thick}"
        self.viewer.layers[1].size = [self.z_thick_list[z_thick], self.point_size, self.point_size]
