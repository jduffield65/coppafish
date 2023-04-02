import os
import numpy as np
import distinctipy
import matplotlib.pyplot as plt
import warnings
import napari
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QDoubleSlider, QRangeSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow, QSlider
from ...setup import Notebook
from ..stitch.diagnostics import shift_info_plot
from scipy.ndimage import affine_transform
from typing import Optional
plt.style.use('dark_background')


def view_transform_distribution(nb: Notebook, transform_type='round', regularised=False):
    # TODO: Add pre and post transform regularisation and do it for the same colorbar
    # TODO: Remove the shift from here and put these in another function that shows the shifts for a particular round or
    #  channel on an xy tile positions before and after regularisation
    # TODO: Add another, more simple diagnostic that just mimcs this but for shifts
    if transform_type == 'round':
        if not regularised:
            transform = nb.register.round_transform[:, nb.basic_info.use_rounds]
    elif transform_type == 'channel':
        transform = nb.register.channel_transform[nb.basic_info.use_channels]

    scale = np.array([transform[:, :, 0, 0].T, transform[:, :, 1, 1].T, transform[:, :, 2, 2].T])
    shift = np.array([transform[:, :, 0, 3].T, transform[:, :, 1, 3].T, transform[:, :, 2, 3].T])

    # Plot shifts on left in order z y x
    plt.subplot(321)
    plt.imshow(shift[0])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Z-Shift', fontsize=20)
    plt.colorbar()

    plt.subplot(323)
    plt.imshow(shift[1])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Y-Shift', fontsize=20)
    plt.colorbar()

    plt.subplot(325)
    plt.imshow(shift[2])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('X-Shift', fontsize=20)
    plt.colorbar()

    # Plot scales on right in order z y x
    plt.subplot(322)
    plt.imshow(scale[0])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Z-Scale', fontsize=20)
    plt.colorbar()

    plt.subplot(324)
    plt.imshow(scale[1])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('Y-Scale', fontsize=20)
    plt.colorbar()

    plt.subplot(326)
    plt.imshow(scale[2])
    plt.xlabel('Tile', fontsize=20)
    plt.ylabel(transform_type, fontsize=20)
    plt.title('X-Scale', fontsize=20)
    plt.colorbar()
    plt.show()


def view_scale_boxplots(nb, transform_type='round'):
    if transform_type == 'round':
        transform = nb.register.round_transform[:, nb.basic_info.use_rounds]
    elif transform_type == 'channel':
        transform = nb.register.channel_transform[nb.basic_info.use_channels]
    scale = np.array([transform[:, :, 0, 0].T, transform[:, :, 1, 1].T, transform[:, :, 2, 2].T])

    plt.subplot(311)
    plt.boxplot(scale[0].T)
    plt.title('Z-Scales', fontsize=20)
    plt.xlabel(transform_type, fontsize=20)
    plt.ylabel('scale', fontsize=20)
    plt.subplot(312)
    plt.boxplot(scale[1].T)
    plt.title('Y-Scales', fontsize=20)
    plt.xlabel(transform_type, fontsize=20)
    plt.ylabel('scale', fontsize=20)
    plt.subplot(313)
    plt.boxplot(scale[2].T)
    plt.title('X-Scales', fontsize=20)
    plt.xlabel(transform_type, fontsize=20)
    plt.ylabel('scale', fontsize=20)


def scale_box_plots(nb: Notebook):
    """
    Function to plot distribution of chromatic aberration scaling amongst tiles for each round and channel.
    Want very similar values for a given channel across all tiles and rounds for each dimension.
    Also expect $y$ and $x$ scaling to be very similar. $z$ scaling different due to unit conversion.

    Args:
        nb: *Notebook* containing the `register` and `register_debug` *NotebookPages*.
    """
    if nb.basic_info.is_3d:
        ndim = 3
        if np.ptp(nb.register.transform[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                               nb.basic_info.use_channels)][:, :, :, 2, 2]) < 1e-5:
            ndim = 2
            warnings.warn("Not showing z-scaling as all are the same")
    else:
        ndim = 2

    fig, ax = plt.subplots(ndim, figsize=(10, 6), sharex=True)
    ax[0].get_shared_y_axes().join(ax[0], ax[1])
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15)
    y_titles = ["Scaling - Y", "Scaling - X", "Scaling - Z"]
    n_use_channels = len(nb.basic_info.use_channels)
    # different boxplot color for each channel
    # Must be distinct from black and white
    channel_colors = distinctipy.get_colors(n_use_channels, [(0, 0, 0), (1, 1, 1)])
    for i in range(ndim):
        box_data = [nb.register.transform[nb.basic_info.use_tiles, r, c, i, i] for c in nb.basic_info.use_channels
                    for r in nb.basic_info.use_rounds]
        bp = ax[i].boxplot(box_data, notch=0, sym='+', patch_artist=True)
        leg_markers = []
        c = -1
        for j in range(len(box_data)):
            if j % n_use_channels == 0:
                c += 1
                leg_markers = leg_markers + [bp['boxes'][j]]
            bp['boxes'][j].set_facecolor(channel_colors[c])
        ax[i].set_ylabel(y_titles[i])

        if i == ndim-1:
            tick_labels = np.tile(nb.basic_info.use_rounds, n_use_channels).tolist()
            leg_labels = nb.basic_info.use_channels
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels)
            ax[i].legend(leg_markers, leg_labels, title='Channel')
            ax[i].set_xlabel('Round')
    ax[0].set_title('Boxplots showing distribution of scalings due to\nchromatic aberration amongst tiles for each '
                    'round and channel')
    plt.show()


class view_affine_shift_info:
    def __init__(self, nb: Notebook, c: Optional[int] = None, outlier: bool = False):
        """
        For all affine transforms to imaging rounds/channels from the reference round computed in the `register` section
        of the pipeline, this plots the values of the shifts, `n_matches` (number of neighbours found) and
        `error` (average distance between neighbours).

        For each round and channel (channel is changed by scrolling with the mouse), there will be 3 plots:

        * y shift vs x shift for all tiles
        * z shift vs x shift for all tiles
        * `n_matches` vs `error` for all tiles

        In each case, the markers in the plots are numbers.
        These numbers indicate the tile the shift was found for.
        The number will be blue if `nb.register_debug.n_matches > nb.register_debug.n_matches_thresh` and red otherwise.

        Args:
            nb: Notebook containing at least the `register` page.
            c: If None, will give option to scroll with mouse to change channel. If specify c, will show just
                one channel with no scrolling.
            outlier: If `True`, will plot shifts from `nb.register_debug.transform_outlier` instead of
                `nb.register.transform`. In this case, only tiles for which
                `nb.register_debug.failed == True` are plotted for each round/channel.
        """
        self.outlier = outlier
        self.nb = nb
        if c is None:
            if self.outlier:
                # only show channels for which there is an outlier shift
                self.channels = np.sort(np.unique(np.where(nb.register_debug.failed)[2]))
                if len(self.channels) == 0:
                    raise ValueError(f"No outlier transforms were computed")
            else:
                self.channels = np.asarray(nb.basic_info.use_channels)
        else:
            self.channels = [c]
        self.n_channels = len(self.channels)
        self.c_ind = 0
        self.c = self.channels[self.c_ind]

        n_cols = len(nb.basic_info.use_rounds)
        if nb.basic_info.is_3d:
            n_rows = 3
        else:
            n_rows = 2
        self.fig, self.ax = plt.subplots(n_rows, n_cols, figsize=(15, 7))
        self.fig.subplots_adjust(hspace=0.4, bottom=0.08, left=0.06, right=0.97, top=0.9)
        self.shift_info = self.get_ax_lims(self.nb, self.channels, self.outlier)
        self.update()
        if self.n_channels > 1:
            self.fig.canvas.mpl_connect('scroll_event', self.z_scroll)
        plt.show()

    @staticmethod
    def get_ax_lims(nb: Notebook, channels, outlier: bool):
        # initialises shift_info with ax limits for each plot
        # Do this because want limits to remain the same as we change color channel.
        if nb.basic_info.is_3d:
            ndim = 3
            z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
        else:
            ndim = 2
        y_lim = np.zeros((ndim, 2))
        x_lim = np.zeros((ndim, 2))

        # same matches/error limits for all rounds as well as all channels
        n_matches = nb.register_debug.n_matches[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                       channels)]
        y_lim[-1, :] = [np.clip(np.min(n_matches) - 100, 0, np.inf), np.max(n_matches) + 100]
        error = nb.register_debug.error[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                       channels)]
        x_lim[-1, :] = [np.clip(np.min(error) - 0.1, 0, np.inf), np.max(error) + 0.1]

        if outlier:
            shifts = nb.register_debug.transform_outlier[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                                   channels)][:, :, :, 3]
        else:
            shifts = nb.register.transform[np.ix_(nb.basic_info.use_tiles, nb.basic_info.use_rounds,
                                                     channels)][:, :, :, 3]
        if ndim == 3:
            shifts[:, :, :, 2] = shifts[:, :, :, 2] / z_scale  # put z shift in units of z-pixels

        shift_info = {}
        for r in range(len(nb.basic_info.use_rounds)):
            name = f'Round {nb.basic_info.use_rounds[r]}'
            shift_info[name] = {}
            shift_info[name]['y_lim'] = y_lim.copy()
            shift_info[name]['x_lim'] = x_lim.copy()
            # 1st plot (Y shift vs X shift)
            shift_info[name]['y_lim'][0] = [np.min(shifts[:, r, :, 0]) - 3, np.max(shifts[:, r, :, 0]) + 3]
            shift_info[name]['x_lim'][0] = [np.min(shifts[:, r, :, 1]) - 3, np.max(shifts[:, r, :, 1]) + 3]
            if ndim == 3:
                # 2nd plot (Z shift vs X shift)
                shift_info[name]['y_lim'][1] = [np.min(shifts[:, r, :, 2]) - 1, np.max(shifts[:, r, :, 2]) + 1]
                shift_info[name]['x_lim'][1] = [np.min(shifts[:, r, :, 1]) - 3, np.max(shifts[:, r, :, 1]) + 3]
        return shift_info

    @staticmethod
    def get_shift_info(shift_info: dict, nb: Notebook, c: int, outlier: bool) -> dict:
        # Updates the shift_info dictionary to pass to shift_info_plot
        if nb.basic_info.is_3d:
            ndim = 3
            z_scale = nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy
        else:
            ndim = 2
        for r in nb.basic_info.use_rounds:
            name = f'Round {r}'
            shift_info[name]['tile'] = nb.basic_info.use_tiles
            if outlier:
                shift_info[name]['shift'] = nb.register_debug.transform_outlier[nb.basic_info.use_tiles, r, c, 3, :ndim]
            else:
                shift_info[name]['shift'] = nb.register.transform[nb.basic_info.use_tiles, r, c, 3, :ndim]
            if ndim == 3:
                # put z-shift in units of z-pixels
                shift_info[name]['shift'][:, 2] = shift_info[name]['shift'][:, 2] / z_scale
            shift_info[name]['n_matches'] = nb.register_debug.n_matches[nb.basic_info.use_tiles, r, c]
            shift_info[name]['n_matches_thresh'] = nb.register_debug.n_matches_thresh[nb.basic_info.use_tiles, r, c]
            if outlier:
                # Set matches to 0 if no outlier transform found so won't plot
                shift_info[name]['n_matches'][np.invert(nb.register_debug.failed[nb.basic_info.use_tiles, r, c])] = 0
            shift_info[name]['error'] = nb.register_debug.error[nb.basic_info.use_tiles, r, c]
        return shift_info

    def update(self):
        # Gets shift_info for current channel and updates plot figure.
        shift_info = self.get_shift_info(self.shift_info, self.nb, self.c, self.outlier)
        for ax in self.ax.flatten():
            ax.cla()
        if self.outlier:
            title_start = "Outlier "
        else:
            title_start = ""
        self.ax = shift_info_plot(shift_info, f"{title_start}Shifts found in register part of pipeline "
                                              f"from round {self.nb.basic_info.anchor_round}, channel "
                                              f"{self.nb.basic_info.anchor_channel} to channel "
                                              f"{self.c} for each round and tile",
                                  fig=self.fig, ax=self.ax, return_ax=True)
        self.ax[0, 0].figure.canvas.draw()

    def z_scroll(self, event):
        # Scroll to change channel shown in plots
        if event.button == 'up':
            self.c_ind = (self.c_ind + 1) % self.n_channels
        else:
            self.c_ind = (self.c_ind - 1) % self.n_channels
        self.c = self.channels[self.c_ind]
        self.update()


class RegistrationViewer:
    def __init__(self, nb: Notebook, t: int = None):
        """
        Function to overlay tile, round and channel with the anchor in napari and view the registration.
        This function is only long because we have to convert images to zyx and the transform to zyx * zyx
        Args:
            nb: Notebook
            t: common tile
        """
        # initialise frequently used variables, attaching those which are otherwise awkward to recalculate to self
        nbp_file, nbp_basic = nb.file_names, nb.basic_info
        use_rounds, use_channels = nbp_basic.use_rounds, nbp_basic.use_channels
        # set default transform to svr transform
        self.transform = nb.register.start_transform
        self.z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
        self.r_ref, self.c_ref = nbp_basic.anchor_round, nb.basic_info.anchor_channel
        self.r_mid = len(use_rounds) // 2
        y_mid, x_mid, z_mid = nbp_basic.tile_centre
        self.new_origin = np.array([z_mid - 5, y_mid - 250, x_mid - 250])

        # Initialise file directories
        self.target_round_image = []
        self.target_channel_image = []
        self.base_image = None
        self.output_dir = os.path.join(nbp_file.output_dir, 'reg_images/')

        # Attach the 2 arguments to the object to be created and a new object for the viewer
        self.nb = nb
        self.viewer = napari.Viewer()

        # Make layer list invisible to remove clutter
        self.viewer.window.qt_viewer.dockLayerList.setVisible(False)
        self.viewer.window.qt_viewer.dockLayerControls.setVisible(False)

        # Now we will create 2 sliders. One will control all the contrast limits simultaneously, the other all anchor
        # images simultaneously.
        self.im_contrast_limits_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.anchor_contrast_limits_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.im_contrast_limits_slider.setRange(0, 256)
        self.anchor_contrast_limits_slider.setRange(0, 256)
        # Set default lower limit to 0 and upper limit to 100
        self.im_contrast_limits_slider.setValue(([0, 100]))
        self.anchor_contrast_limits_slider.setValue([0, 100])
        # Now we run a method that sets these contrast limits using napari
        # Create sliders!
        self.viewer.window.add_dock_widget(self.im_contrast_limits_slider, area="left", name='Imaging Contrast')
        self.viewer.window.add_dock_widget(self.anchor_contrast_limits_slider, area="left", name='Anchor Contrast')
        # Now create events that will recognise when someone has changed slider values
        self.anchor_contrast_limits_slider.valueChanged.connect(lambda x:
                                                                self.change_anchor_layer_contrast(x[0], x[1]))
        self.im_contrast_limits_slider.valueChanged.connect(lambda x: self.change_imaging_layer_contrast(x[0], x[1]))

        # Add buttons to change between registration methods
        self.method_buttons = ButtonMethodWindow('SVR')
        # I think this allows us to connect button status with the buttons in the viewer
        self.method_buttons.button_icp.clicked.connect(self.button_icp_clicked)
        self.method_buttons.button_svr.clicked.connect(self.button_svr_clicked)
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.method_buttons, area="left", name='Method')

        # Add buttons to select different tiles. Involves initialising variables use_tiles and tilepos
        tilepos_xy = np.roll(self.nb.basic_info.tilepos_yx, shift=1, axis=1)
        # Invert y as y goes downwards in the set geometry func
        num_rows = np.max(tilepos_xy[:, 1])
        tilepos_xy[:, 1] = num_rows - tilepos_xy[:, 1]
        # get use tiles
        use_tiles = self.nb.basic_info.use_tiles
        # If no tile provided then default to the first tile in use
        if t is None:
            t = use_tiles[0]
        self.tile = t
        self.tile_buttons = ButtonTileWindow(tile_pos_xy=tilepos_xy, use_tiles=use_tiles, active_button=self.tile)
        # We need to connect all these buttons to a single function which updates the tile in the same way
        for tile in range(len(tilepos_xy)):
            self.tile_buttons.__getattribute__(str(tile)).clicked.connect(lambda x=tile: self.tile_button_clicked(x))
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.tile_buttons, area="left", name='Tiles')

        # Get target images and anchor image
        self.get_images()

        # Plot images
        self.plot_images()

        # Set default contrast limits. Have to do this here as the images are only defined now
        self.change_anchor_layer_contrast(self.anchor_contrast_limits_slider.value()[0],
                                          self.anchor_contrast_limits_slider.value()[1])
        self.change_imaging_layer_contrast(self.im_contrast_limits_slider.value()[0],
                                           self.im_contrast_limits_slider.value()[1])

        napari.run()

    def change_anchor_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(0, 32, 2):
            self.viewer.layers[i].contrast_limits = [low, high]

    def change_imaging_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(1, 32, 2):
            self.viewer.layers[i].contrast_limits = [low, high]

    def button_svr_clicked(self):
        # Only allow one button pressed
        # Below does nothing if method is already svr and updates plot otherwise
        if self.method_buttons.method == 'SVR':
            self.method_buttons.button_svr.setChecked(True)
            self.method_buttons.button_icp.setChecked(False)
        else:
            self.method_buttons.button_svr.setChecked(True)
            self.method_buttons.button_icp.setChecked(False)
            self.method_buttons.method = 'SVR'
            # Because method has changed, also need to change transforms
            # Update set of transforms
            self.transform = self.nb.register.start_transform
            self.update_plot()

    def button_icp_clicked(self):
        # Only allow one button pressed
        # Below does nothing if method is already icp and updates plot otherwise
        if self.method_buttons.method == 'ICP':
            self.method_buttons.button_icp.setChecked(True)
            self.method_buttons.button_svr.setChecked(False)
        else:
            self.method_buttons.button_icp.setChecked(True)
            self.method_buttons.button_svr.setChecked(False)
            self.method_buttons.method = 'ICP'
            # Because method has changed, also need to change transforms
            # Update set of transforms
            self.transform = self.nb.register.transform
            self.update_plot()

    def tile_button_clicked(self, t):
        # This method should change the image iff self.tile_buttons.tile has changed
        use_tiles = self.nb.basic_info.use_tiles
        if self.tile_buttons.tile == str(t):
            for tile in use_tiles:
                self.tile_buttons.__getattribute__(str(tile)).setChecked(tile == t)
        else:
            for tile in use_tiles:
                self.tile_buttons.__getattribute__(str(tile)).setChecked(tile == t)
            # Because tile under consideration changed, need to update plot and update tile_buttons.tile parameter
            if t in use_tiles:
                self.tile_buttons.tile = str(t)
                self.update_plot()

    def update_plot(self):
        # Updates plot if tile or method has been changed
        # First get num rounds and channels
        n_rounds, n_channels = len(self.nb.basic_info.use_rounds), len(self.nb.basic_info.use_channels)
        # Update the images, we reload the anchor image even when it has not been changed, this should not be too slow
        self.get_images()
        self.plot_images()

    def get_images(self):
        # reset initial target image lists to empty lists
        use_rounds, use_channels = self.nb.basic_info.use_rounds, self.nb.basic_info.use_channels
        self.target_round_image, self.target_channel_image = [], []
        t = self.tile
        # populate target arrays
        for r in use_rounds:
            file = 't'+str(t) + 'r'+str(r) + 'c'+str(self.c_ref)+'.npy'
            affine = change_basis(self.transform[t, r, self.c_ref], new_origin=self.new_origin, z_scale=self.z_scale)
            # Reset the spline interpolation order to 1 to speed things up
            self.target_round_image.append(affine_transform(np.load(os.path.join(self.output_dir, file)),
                                                            affine, order=1))

        for c in use_channels:
            file = 't' + str(t) + 'r' + str(self.r_mid) + 'c' + str(c) + '.npy'
            affine = change_basis(self.transform[t, self.r_mid, c], new_origin=self.new_origin, z_scale=self.z_scale)
            self.target_channel_image.append(affine_transform(np.load(os.path.join(self.output_dir, file)),
                                                              affine, order=1))
        # populate anchor image
        anchor_file = 't' + str(t) + 'r' + str(self.r_ref) + 'c' + str(self.c_ref) + '.npy'
        self.base_image = np.load(os.path.join(self.output_dir, anchor_file))

    def plot_images(self):
        use_rounds, use_channels = self.nb.basic_info.use_rounds, self.nb.basic_info.use_channels

        # We will add a point on top of each image and add features to it
        features = {'round': np.repeat(np.append(use_rounds, np.ones(len(use_channels)) * self.r_mid), 10).astype(int),
                    'channel': np.repeat(np.append(np.ones(len(use_rounds)) * self.c_ref, use_channels), 10).astype(
                        int)}

        # Define text
        text = {
            'string': 'Round {round} Channel {channel}',
            'size': 20,
            'color': 'white'}

        # Now go on to define point coords
        points = []

        for r in use_rounds:
            self.viewer.add_image(self.base_image, blending='additive', colormap='red', translate=[0, 0, 1_000 * r],
                                  name='Anchor')
            self.viewer.add_image(self.target_round_image[r], blending='additive', colormap='green',
                                  translate=[0, 0, 1_000 * r], name='Round ' + str(r) + ', Channel ' + str(self.c_ref))
            for z in range(10):
                points.append([z, -50, 250 + 1_000 * r])

        for c in range(len(use_channels)):
            self.viewer.add_image(self.base_image, blending='additive', colormap='red', translate=[0, 1_000, 1_000 * c],
                                  name='Anchor')
            self.viewer.add_image(self.target_channel_image[c], blending='additive', colormap='green',
                                  translate=[0, 1_000, 1_000 * c],
                                  name='Round ' + str(self.r_mid) + ', Channel ' + str(use_channels[c]))
            for z in range(10):
                points.append([z, 950, 250 + 1_000 * c])

        # Add text to image
        self.viewer.add_points(np.array(points), features=features, text=text, size=1)


class ButtonMethodWindow(QMainWindow):
    def __init__(self, active_button: str = 'SVR'):
        super().__init__()
        self.button_svr = QPushButton('SVR', self)
        self.button_svr.setCheckable(True)
        self.button_svr.setGeometry(75, 2, 50, 28)  # left, top, width, height

        self.button_icp = QPushButton('ICP', self)
        self.button_icp.setCheckable(True)
        self.button_icp.setGeometry(140, 2, 50, 28)  # left, top, width, height
        if active_button.lower() == 'icp':
            # Initially, show sub vol regression registration
            self.button_icp.setChecked(True)
            self.method = 'ICP'
        elif active_button.lower() == 'svr':
            self.button_svr.setChecked(True)
            self.method = 'SVR'
        else:
            raise ValueError(f"active_button should be 'SVR' or 'ICP' but {active_button} was given.")


class ButtonTileWindow(QMainWindow):
    def __init__(self, tile_pos_xy: np.ndarray, use_tiles: list, active_button: 0):
        super().__init__()
        # Loop through tiles, putting them in location as specified by tile pos xy
        for t in range(len(tile_pos_xy)):
            # Create a button for each tile
            button = QPushButton(str(t), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(t in use_tiles)
            button.setGeometry(tile_pos_xy[t, 0] * 70, tile_pos_xy[t, 1] * 40, 50, 28)
            # set active button as checked
            if active_button == t:
                button.setChecked(True)
                self.tile = t
            # Set button color = grey when hovering over
            # set colour of tiles in use to blue amd not in use to red
            if t in use_tiles:
                button.setStyleSheet("QPushButton"
                                     "{"
                                     "background-color : rgb(135, 206, 250);"
                                     "}"
                                     "QPushButton::hover"
                                     "{"
                                     "background-color : lightgrey;"
                                     "}"
                                     "QPushButton::pressed"
                                     "{"
                                     "background-color : white;"
                                     "}")
            else:
                button.setStyleSheet("QPushButton"
                                     "{"
                                     "background-color : rgb(240, 128, 128);"
                                     "}"
                                     "QPushButton::hover"
                                     "{"
                                     "background-color : lightgrey;"
                                     "}"
                                     "QPushButton::pressed"
                                     "{"
                                     "background-color : white;"
                                     "}")
            # Finally add this button as an attribute to self
            self.__setattr__(str(t), button)


# TODO: Change format of this to make it more similar to notebook outputs
def view_regression_scatter(shift, position, transform):
    """
    view 3 scatter plots for each data set shift vs positions
    Args:
        shift: z_sv x y_sv x x_sv x 3 array which of shifts in zyx format
        position: z_sv x y_sv x x_sv x 3 array which of positions in zyx format
        transform: 3 x 4 affine transform obtained by previous robust regression
    """

    shift = shift.reshape((shift.shape[0] * shift.shape[1] * shift.shape[2], 3)).T
    position = position.reshape((position.shape[0] * position.shape[1] * position.shape[2], 3)).T

    pos_shape = np.array(position.shape).astype(int)
    rand = 100 * np.random.rand(pos_shape[0], pos_shape[1]) - 100
    position = position + rand

    z_range = np.arange(np.min(position[0]), np.max(position[0]))
    yx_range = np.arange(np.min(position[1]), np.max(position[1]))

    plt.subplot(1, 3, 1)
    plt.scatter(position[0], shift[0], alpha=1e2/shift.shape[1])
    plt.plot(z_range, (transform[0, 0] - 1) * z_range + transform[0,3])
    plt.title('Z-Shifts vs Z-Positions')

    plt.subplot(1, 3, 2)
    plt.scatter(position[1], shift[1], alpha=1e2 / shift.shape[1])
    plt.plot(yx_range, (transform[1, 1] - 1) * yx_range + transform[1, 3])
    plt.title('Y-Shifts vs Y-Positions')

    plt.subplot(1, 3, 3)
    plt.scatter(position[2], shift[2], alpha=1e2 / shift.shape[1])
    plt.plot(yx_range, (transform[2, 2] - 1) * yx_range + transform[2, 3])
    plt.title('X-Shifts vs X-Positions')

    plt.show()


def change_basis(A, new_origin, z_scale):
    """
    Takes in 4 x 3 yxz * yxz transform where z coord is in xy pixels and convert to 4 x 4 zyx * zyx where final
    Args:
        A: 4 x 3 yxz * yxz transform
        new_origin: new origin (zyx)
        z_scale: pixel_size_z/pixel_size_xy

    """
    # Transform saved as yxz * yxz but needs to be zyx * zyx. Convert this to something napari will understand. I think
    # this includes making the shift the final column as opposed to our convention of making the shift the final row
    affine_transform = np.vstack((A.T, np.array([0, 0, 0, 1])))

    row_shuffler = np.zeros((4, 4))
    row_shuffler[0, 1] = 1
    row_shuffler[1, 2] = 1
    row_shuffler[2, 0] = 1
    row_shuffler[3, 3] = 1

    # Now compute the affine transform, in the new basis
    affine_transform = np.linalg.inv(row_shuffler) @ affine_transform @ row_shuffler

    # z shift needs to be converted to z-pixels as opposed to yx
    affine_transform[0, 3] = affine_transform[0, 3] / z_scale

    # also add new origin conversion for shift
    affine_transform[:3, 3] += (affine_transform[:3, :3] - np.eye(3)) @ new_origin

    return affine_transform
