import os
import numpy as np
import matplotlib.pyplot as plt
import napari
from qtpy.QtCore import Qt
from superqt import QDoubleRangeSlider, QDoubleSlider, QRangeSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow, QSlider
from ...setup import Notebook, NotebookPage
from coppafish.register.preprocessing import change_basis, stack_images, create_shift_images
from coppafish.register.base import huber_regression
from scipy.ndimage import affine_transform
plt.style.use('dark_background')


# there are 3 parts of the registration pipeline:
# 1. SVR
# 2. Cross tile outlier removal
# 3. ICP
# Above each of these viewers we will plot a number which shows which part it refers to

# 1 and 3
class RegistrationViewer:
    # TODO: FIx bug for switching between SVR and ICP
    # TODO: Add key binding to show pearson r distributions
    # TODO: Add buttons for channel and round regressions
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

        # Add buttons to show round regression
        # self.round_buttons = ButtonRoundWindow(use_rounds=nbp_basic.use_rounds)
        # We need to connect all these buttons to a single function which plots image in the same way
        # for r in use_rounds:
        #     self.round_buttons.__getattribute__(str(r)).clicked.connect(self.round_button_clicked(r))
        # Add these buttons as widgets in napari viewer
        # self.viewer.window.add_dock_widget(self.round_buttons, area="left", name='Round Regression')

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
        # Store a copy of the working tile in the RegistrationViewer
        self.tile = t

        # Now create tile_buttons
        self.tile_buttons = ButtonTileWindow(tile_pos_xy=tilepos_xy, use_tiles=use_tiles, active_button=self.tile)
        for tile in use_tiles:
            # Now connect the button associated with tile t to a function that activates t and deactivates all else
            self.tile_buttons.__getattribute__(str(tile)).clicked.connect(self.create_tile_slot(tile))
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.tile_buttons, area="left", name='Tiles', add_vertical_stretch=False)

        # Create round_buttons
        self.round_buttons = ButtonRoundWindow(self.nb.basic_info.use_rounds)
        for rnd in use_rounds:
            # Now connect the button associated with tile t to a function that activates t and deactivates all else
            self.round_buttons.__getattribute__(str(rnd)).clicked.connect(self.create_round_slot(rnd))
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.round_buttons, area="left", name='Round Regression',
                                           add_vertical_stretch=False)

        # Create channel_buttons
        self.channel_buttons = ButtonChannelWindow(self.nb.basic_info.use_channels)
        for c in use_channels:
            # Now connect the button associated with tile t to a function that activates t and deactivates all else
            self.channel_buttons.__getattribute__(str(c)).clicked.connect(self.create_channel_slot(c))
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.channel_buttons, area="left", name='Channel Regression',
                                           add_vertical_stretch=False)

        # Get target images and anchor image
        self.get_images()

        # Plot images
        self.plot_images()

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

    def create_round_slot(self, r):

        def round_button_clicked():
            use_rounds = self.nb.basic_info.use_rounds
            for rnd in use_rounds:
                self.round_buttons.__getattribute__(str(rnd)).setChecked(rnd == r)
            # We don't need to update the plot, we just need to call the viewing function
            view_regression_scatter(shift=self.nb.register.round_shift[self.tile, r],
                                    position=self.nb.register.round_position[self.tile, r],
                                    transform=self.nb.register.round_transform[self.tile, r])
        return round_button_clicked

    def create_tile_slot(self, t):

        def tile_button_clicked():
            # We're going to connect each button str(t) to a function that sets checked str(t) and nothing else
            # Also sets self.tile = t
            use_tiles = self.nb.basic_info.use_tiles
            for tile in use_tiles:
                self.tile_buttons.__getattribute__(str(tile)).setChecked(tile == t)
            self.tile = t
            self.update_plot()

        return tile_button_clicked

    def create_channel_slot(self, c):

        def channel_button_clicked():
            use_channels = self.nb.basic_info.use_channels
            for chan in use_channels:
                self.channel_buttons.__getattribute__(str(chan)).setChecked(chan == c)
            # We don't need to update the plot, we just need to call the viewing function
            view_regression_scatter(shift=self.nb.register.channel_shift[self.tile, c],
                                    position=self.nb.register.channel_position[self.tile, c],
                                    transform=self.nb.register.channel_transform[self.tile, c])
        return channel_button_clicked

    def update_plot(self):
        # Updates plot if tile or method has been changed
        # Update the images, we reload the anchor image even when it has not been changed, this should not be too slow
        self.clear_images()
        self.get_images()
        self.plot_images()

    def clear_images(self):
        # Function to clear all images currently in use
        n_images = len(self.viewer.layers)
        for i in range(n_images):
            del self.viewer.layers[0]

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
                                  name='Anchor', contrast_limits=[0, 100])
            self.viewer.add_image(self.target_round_image[r], blending='additive', colormap='green',
                                  translate=[0, 0, 1_000 * r], name='Round ' + str(r) + ', Channel ' + str(self.c_ref),
                                  contrast_limits=[0, 100])
            for z in range(10):
                points.append([z, -50, 250 + 1_000 * r])

        for c in range(len(use_channels)):
            self.viewer.add_image(self.base_image, blending='additive', colormap='red', translate=[0, 1_000, 1_000 * c],
                                  name='Anchor', contrast_limits=[0, 100])
            self.viewer.add_image(self.target_channel_image[c], blending='additive', colormap='green',
                                  translate=[0, 1_000, 1_000 * c],
                                  name='Round ' + str(self.r_mid) + ', Channel ' + str(use_channels[c]),
                                  contrast_limits=[0, 100])
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


class ButtonRoundWindow(QMainWindow):
    def __init__(self, use_rounds: list):
        super().__init__()
        # Loop through tiles, putting them in location as specified by tile pos xy
        for r in use_rounds:
            # Create a button for each tile
            button = QPushButton(str(r), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(True)
            button.setGeometry(r * 70, 40, 50, 28)
            # Set button color = grey when hovering over
            # set colour of tiles in use to blue amd not in use to red
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
            # Finally add this button as an attribute to self
            self.__setattr__(str(r), button)
            self.round_regression = None


class ButtonChannelWindow(QMainWindow):
    def __init__(self, use_channels: list):
        super().__init__()
        # Loop through tiles, putting them in location as specified by tile pos xy
        for c in range(len(use_channels)):
            # Create a button for each tile
            button = QPushButton(str(use_channels[c]), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(True)
            button.setGeometry(c * 70, 40, 50, 28)
            # Set button color = grey when hovering over
            # set colour of tiles in use to blue amd not in use to red
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
            # Finally add this button as an attribute to self
            self.__setattr__(str(use_channels[c]), button)


# 1
def view_regression_scatter(shift, position, transform=None):
    """
    view 3 scatter plots for each data set shift vs positions
    Args:
        shift: (z_sv x y_sv x x_sv) x 3 array which of shifts in zyx format
        position: (z_sv x y_sv x x_sv) x 3 array which of positions in zyx format
        transform: 3 x 4 affine transform obtained by previous robust regression
        save_loc: save location if applicable
    """
    shift = np.reshape(shift, (shift.shape[0] * shift.shape[1] * shift.shape[2], 3))
    position = np.reshape(position, (position.shape[0] * position.shape[1] * position.shape[2], 3))
    shift = shift.T
    position = position.T

    z_range = np.arange(np.min(position[0]), np.max(position[0]))
    yx_range = np.arange(np.min(position[1]), np.max(position[1]))

    plt.subplot(1, 3, 1)
    plt.scatter(position[0], shift[0], alpha=0.1)
    plt.title('Z-Shifts vs Z-Positions')

    plt.subplot(1, 3, 2)
    plt.scatter(position[1], shift[1], alpha=0.1)
    plt.title('Y-Shifts vs Y-Positions')

    plt.subplot(1, 3, 3)
    plt.scatter(position[2], shift[2], alpha=0.1)
    plt.title('X-Shifts vs X-Positions')

    # If transform not none then plot line
    if transform is not None:
        plt.subplot(1, 3, 1)
        plt.plot(z_range, (transform[0, 0] - 1) * z_range + transform[0, 3])

        plt.subplot(1, 3, 2)
        plt.plot(yx_range, (transform[1, 1] - 1) * yx_range + transform[1, 3])

        plt.subplot(1, 3, 3)
        plt.plot(yx_range, (transform[2, 2] - 1) * yx_range + transform[2, 3])

    plt.show()


# 1
def view_pearson_hists(nbp_register_debug, nbp_basic, t, thresh=0.4, num_bins=30):
    """
    function to view histogram of correlation coefficients for all subvol shifts of a particular round/channel.
    Args:
        nbp_register_debug:
        nbp_basic:
        thresh: (float) threshold of r value to get used in robust regression
        num_bins: int number of bins in the histogram
        t: int tile under consideration
    """
    round_corr, channel_corr = nbp_register_debug.round_corr[t], nbp_register_debug.channel_corr[t]
    n_rounds, n_channels_use = nbp_basic.n_rounds, len(nbp_basic.use_channels)
    use_channels = nbp_basic.use_channels
    cols = max(n_rounds, n_channels_use)

    for r in range(n_rounds):
        plt.subplot(2, cols, r + 1)
        counts, _ = np.histogram(round_corr[r], np.linspace(0, 1, num_bins))
        plt.hist(round_corr[r], bins=np.linspace(0, 1, num_bins))
        plt.vlines(x=thresh, ymin=0, ymax=np.max(counts), colors='r')
        plt.title('Quality of sub-volume shifts for tile ' + str(t) + ', round ' + str(r) +
                  '\n Inilier proportion = ' + str(
            round(100 * sum(round_corr[r] > thresh) / round_corr.shape[1], 2)) + '%')

    for c in range(n_channels_use):
        plt.subplot(2, cols, cols + c + 1)
        counts, _ = np.histogram(channel_corr[use_channels[c]], np.linspace(0, 1, num_bins))
        plt.hist(channel_corr[use_channels[c]], bins=np.linspace(0, 1, num_bins))
        plt.vlines(x=thresh, ymin=0, ymax=np.max(counts), colors='r')
        plt.title('Quality of sub-volume shifts for tile ' + str(t) + ', channel ' + str(use_channels[c]) +
                  '\n Inilier proportion = ' + str(
            round(100 * sum(channel_corr[use_channels[c]] > thresh) / channel_corr.shape[1], 2)) + '%')

    plt.show()
    plt.suptitle('Similarity distributions for all subvolume shifts')


# 1
def view_pearson_colourmap(nbp_register_debug, nbp_basic, t):
    """
    function to view colourmap of correlation coefficients for all subvol shifts for all channels and rounds.

    Args:
        nbp_register_debug: register debug notebook page
        nbp_basic:register basic info notebook page
        t: int tile under consideration
    """
    # initialise frequently used variables
    round_corr, channel_corr = nbp_register_debug.round_corr[t], nbp_register_debug.channel_corr[t]
    use_channels = nbp_basic.use_channels
    # Replace 0 with nans so they get plotted as black
    round_corr[round_corr == 0] = np.nan
    channel_corr[channel_corr == 0] = np.nan

    # plot round correlation and tile correlation
    plt.subplot(1, 2, 1)
    plt.imshow(round_corr)
    plt.subplot(1, 2, 2)
    plt.imshow(channel_corr[:, use_channels])

    plt.suptitle('Similarity distributions for all subvolume shifts')


# 1
def view_pearson_colourmap_spatial(nbp_register_debug, config, t, round: bool, index: int):
    """
    function to view colourmap of correlation coefficients for all subvol shifts for a single t, r, c.

    Args:
        nbp_register_debug: register debug notebook page
        config: register page of config dict
        t: int tile under consideration
        round: True if round, False if channel
        index: int rc in question
    """

    # initialise frequently used variables
    if round:
        corr = nbp_register_debug.round_corr[t, index]
    else:
        corr = nbp_register_debug.channel_corr[t, index]

    corr[corr == 0] = np.nan
    z_subvols, y_subvols, x_subvols = config['z_subvols'], config['y_subvols'], config['x_subvols']

    fig, axes = plt.subplots(nrows=1, ncols=z_subvols)
    for z in range(z_subvols):
        ax = axes[0, z]
        im = ax.imshow(np.reshape(corr[z * y_subvols * x_subvols: (z + 1) * x_subvols * y_subvols],
                                  (y_subvols, x_subvols)), vmin=0, vmax=1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


# 2
def shift_vector_field(nbp_register_debug: NotebookPage, nbp_basic: NotebookPage, round: bool, config: dict):
    """
    Function to plot vector fields of predicted shifts vs shifts to see if we classify a shift as an outlier.
    Args:
        nbp_register_debug: register debug notebook page
        nbp_basic: basic info notebook page
        round: Boolean indicating whether we are looking at round outlier removal, True if r, False if c
        config: Register page of config dictionary
    """
    use_tiles = nbp_basic.use_tiles
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]

    # Load in shift
    if round:
        shift = nbp_register_debug.round_transform_unregularised[use_tiles, :, :, 3]
    else:
        shift = nbp_register_debug.channel_transform_unregularised[use_tiles, :, :, 3]

    # record number of rounds/channels, tiles and initialise predicted shift
    n_t, n_rc = shift.shape[0], shift.shape[1]
    tilepos_yx_pad = np.vstack((tilepos_yx.T, np.ones(n_t))).T
    predicted_shift = np.zeros_like(shift)

    fig, axes = plt.subplots(nrows=3, ncols=n_rc)
    for elem in range(n_rc):
        # generate predicted shift for this r/c via huber regression
        transform = huber_regression(shift[elem], tilepos_yx)
        predicted_shift[:, elem] = tilepos_yx_pad @ transform.T

        # plot the predicted yx shift vs actual yx shift in row 0
        ax = axes[0, elem]
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 10 * predicted_shift[:, elem, 2],
                  10 * predicted_shift[:, elem, 1], color='b')
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 10 * shift[:, elem, 2], 10 * shift[:, elem, 1], color='r')
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # plot the predicted z shift vs actual z shift in row 1
        ax = axes[1, elem]
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, 10 * predicted_shift[:, elem, 0], color='b')
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, 10 * shift[:, elem, 2], color='r')
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # Plot image of norms of residuals at each tile in ro2 3
        ax = axes[2, elem]
        diff = make_residual_plot(residual=np.linalg.norm(predicted_shift[:, elem] - shift[:, elem]),
                                  nbp_basic=nbp_basic)
        outlier = np.argwhere(diff > config['residual_threshold'])
        n_outliers = outlier.shape[0]
        im = ax.imshow(diff, vmin=0, vmax=10)
        # Now highlight in red the outlier pixels
        for pixel in range(n_outliers):
            rectangle = plt.Rectangle(outlier[pixel], 1, 1, fill='false', ec='r', linestyle=':', lw=4)
            ax.add_patch(rectangle)
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    # Add colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


# 2
def zyx_shift_image(nbp_register: NotebookPage, nbp_register_debug: NotebookPage, nbp_basic: NotebookPage, round: bool):
    """
        Function to plot vector fields of predicted shifts vs shifts to see if we classify a shift as an outlier.
        Args:
            nbp_register: register notebook page
            nbp_register_debug: register debug notebook page
            nbp_basic: basic info notebook page
            round: Boolean indicating whether we are looking at round outlier removal, True if r, False if c
    """
    use_tiles = nbp_basic.use_tiles
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]

    # Load in shift
    if round:
        shift_raw = nbp_register_debug.round_transform_unregularised[use_tiles, :, :, 3]
        shift = nbp_register.round_transform[use_tiles, :, :, 3]
    else:
        shift_raw = nbp_register_debug.channel_transform_unregularised[use_tiles, :, :, 3]
        shift = nbp_register.channel_transform[use_tiles, :, :, 3]

    n_t, n_rc = shift.shape[0], shift.shape[1]

    fig, axes = plt.subplots(nrows=3, ncols=n_rc)
    for elem in range(n_rc):
        im_raw = create_shift_images(shift_raw[:, elem], tilepos_yx)
        im = create_shift_images(shift[:, elem], tilepos_yx)
        for coord in range(3):
            ax = axes[coord, elem]
            coord_im_stacked = stack_images(im_raw[coord], im[coord])
            ax.imshow(coord_im_stacked, vmin=np.min(shift_raw[:, :, :, coord]), vmax=np.max(shift_raw[:, :, :, coord]))
            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    fig.canvas.draw()
    plt.show()


# 2
def make_residual_plot(residual, nbp_basic):
    """
    generate image of residuals along with their tile positions
    Args:
        residual: n_tiles_use x 1 list of residuals
        nbp_basic: basic info notebook page
    """
    # Initialise frequently used variables
    use_tiles = nbp_basic.use_tiles
    tilepos_yx = nbp_basic.tilepos_yx
    n_rows, n_cols = np.max(tilepos_yx[:, 0]) + 1, np.max(tilepos_yx[:, 1]) + 1
    tilepos_yx = tilepos_yx[nbp_basic.use_tiles]
    diff = np.zeros((n_rows, n_cols))

    for t in use_tiles:
        diff[tilepos_yx[t, 1], tilepos_yx[t, 0]] = residual[t]

    diff = np.flip(diff.T, axis=-1)

    return diff


# 2
def view_round_scales(nbp_register_debug: NotebookPage, nbp_basic: NotebookPage):
    """
    view scale parameters for the round outlier removals
    Args:
        nbp_register_debug: register debug notebook page
        nbp_basic : basic ingo notebook page
    """
    use_tiles = nbp_basic.use_tiles
    # Extract raw scales
    z_scale = nbp_register_debug.round_transform_unregularised[use_tiles, :, 0, 0]
    y_scale = nbp_register_debug.round_transform_unregularised[use_tiles, :, 1, 1]
    x_scale = nbp_register_debug.round_transform_unregularised[use_tiles, :, 2, 2]
    n_tiles_use, n_rounds = z_scale.shape[0], z_scale.shape[1]
    
    # Plot box plots
    plt.subplot(3, 1, 1)
    plt.scatter(np.tile(np.arange(n_rounds), n_tiles_use), np.reshape(z_scale, (n_tiles_use * n_rounds)),
                c='w', marker='x')
    plt.plot(np.arange(n_rounds), np.percentile(z_scale, 25, axis=0), 'c:')
    plt.plot(np.arange(n_rounds), np.percentile(z_scale, 50, axis=0), 'r:')
    plt.plot(np.arange(n_rounds), np.percentile(z_scale, 75, axis=0), 'c:')

    plt.subplot(3, 1, 2)
    plt.scatter(x=np.tile(np.arange(n_rounds), n_tiles_use),
                y=np.reshape(y_scale, (n_tiles_use * n_rounds)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_rounds), 0.999 * np.ones(n_rounds), 'c:')
    plt.plot(np.arange(n_rounds), np.ones(n_rounds), 'r:')
    plt.plot(np.arange(n_rounds), 1.001 * np.ones(n_rounds), 'c:')

    plt.subplot(3, 1, 3)
    plt.scatter(x=np.tile(np.arange(n_rounds), n_tiles_use),
                y=np.reshape(x_scale, (n_tiles_use * n_rounds)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_rounds), 0.999 * np.ones(n_rounds), 'c:')
    plt.plot(np.arange(n_rounds), np.ones(n_rounds), 'r:')
    plt.plot(np.arange(n_rounds), 1.001 * np.ones(n_rounds), 'c:')
    plt.show()


# 2
def view_channel_scales(nbp_register_debug: NotebookPage, nbp_basic: NotebookPage):
    """
    view scale parameters for the round outlier removals
    Args:
        nbp_register_debug: register debug notebook page
        nbp_basic : basic ingo notebook page
    """
    use_tiles = nbp_basic.use_tiles
    use_channels = nbp_basic.use_channels
    # Extract raw scales
    z_scale = nbp_register_debug.channel_transform_unregularised[use_tiles, use_channels, 0, 0]
    y_scale = nbp_register_debug.channel_transform_unregularised[use_tiles, use_channels, 1, 1]
    x_scale = nbp_register_debug.channel_transform_unregularised[use_tiles, use_channels, 2, 2]
    n_tiles_use, n_channels_use = z_scale.shape[0], z_scale.shape[1]

    # Plot box plots
    plt.subplot(3, 1, 1)
    plt.scatter(np.tile(np.arange(n_channels_use), n_tiles_use), np.reshape(z_scale, (n_tiles_use * n_channels_use)),
                c='w', marker='x')
    plt.plot(np.arange(n_channels_use), 0.99 * np.ones(n_channels_use), 'c:')
    plt.plot(np.arange(n_channels_use), np.ones(n_channels_use), 'r:')
    plt.plot(np.arange(n_channels_use), 1.01 * np.ones(n_channels_use), 'c:')

    plt.subplot(3, 1, 2)
    plt.scatter(x=np.tile(np.arange(n_channels_use), n_tiles_use),
                y=np.reshape(y_scale, (n_tiles_use * n_channels_use)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 25, axis=0), 'c:')
    plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 50, axis=0), 'r:')
    plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 75, axis=0), 'c:')

    plt.subplot(3, 1, 3)
    plt.scatter(x=np.tile(np.arange(n_channels_use), n_tiles_use),
                y=np.reshape(x_scale, (n_tiles_use * n_channels_use)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 25, axis=0), 'c:')
    plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 50, axis=0), 'r:')
    plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 75, axis=0), 'c:')
    plt.show()


def view_icp_n_matches(nbp_register_debug: NotebookPage, nbp_basic: NotebookPage,  t):
    """
    Plots simple num matches against iterations
    Args:
        nbp_register_debug: register debug notebook page
        nbp_basic: basic infor page of nb
        t: tile
    """
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_matches = nbp_register_debug.n_matches[t, use_rounds, use_channels]
    n_iters = n_matches.shape[2]

    fig, axes = plt.subplots(len(use_rounds), len(use_channels))
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            ax = axes[r, c]
            ax.plot(np.arange(n_iters), n_matches[use_rounds[r], use_channels[c]])
    plt.show()