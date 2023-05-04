import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import napari
from qtpy.QtCore import Qt
from superqt import QRangeSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow
import matplotlib.gridspec as gridspec
from ...setup import Notebook
from coppafish.register.preprocessing import change_basis, n_matches_to_frac_matches, yxz_to_zyx_affine, \
    compose_affine, invert_affine
from coppafish.register.base import huber_regression
from scipy.ndimage import affine_transform
plt.style.use('dark_background')


# there are 3 parts of the registration pipeline:
# 1. SVR
# 2. Cross tile outlier removal
# 3. ICP
# Above each of these viewers we will plot a number which shows which part it refers to


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
        self.transform = nb.register.transform
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
        # This allows us to link clickng to slot functions
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
        # Store a copy of the working tile in the RegistrationViewer
        self.tile = t

        # Now create tile_buttons
        self.tile_buttons = ButtonTileWindow(tile_pos_xy=tilepos_xy, use_tiles=use_tiles, active_button=self.tile)
        for tile in use_tiles:
            # Now connect the button associated with tile t to a function that activates t and deactivates all else
            self.tile_buttons.__getattribute__(str(tile)).clicked.connect(self.create_tile_slot(tile))
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.tile_buttons, area="left", name='Tiles', add_vertical_stretch=False)

        # We want to create a single napari widget containing buttons which for each round and channel

        # Create all buttons for SVR
        self.svr_buttons = ButtonSVRWindow(self.nb.basic_info.use_rounds, self.nb.basic_info.use_channels)
        # now we begin connecting buttons to functions
        # round buttons
        for rnd in use_rounds:
            # now connect this to a slot that will activate the round regression
            self.svr_buttons.__getattribute__('R'+str(rnd)).clicked.connect(self.create_round_slot(rnd))
        # channel buttons
        for c in use_channels:
            # now connect this to a slot that will activate the channel regression
            self.svr_buttons.__getattribute__('C'+str(c)).clicked.connect(self.create_channel_slot(c))
        # Add buttons for correlation coefficients for both hist or cmap
        self.svr_buttons.pearson_hist.clicked.connect(self.button_pearson_hist_clicked)
        self.svr_buttons.pearson_cmap.clicked.connect(self.button_pearson_cmap_clicked)
        # add buttons for spatial correlation coefficients for rounds or channels
        self.svr_buttons.pearson_spatial_round.clicked.connect(self.button_pearson_spatial_round_clicked)
        self.svr_buttons.pearson_spatial_channel.clicked.connect(self.button_pearson_spatial_channel_clicked)
        # Finally, add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.svr_buttons, area="left", name='SVR Diagnostics',
                                           add_vertical_stretch=False)

        # Create a single widget containing buttons for cross tile outlier removal
        self.outlier_buttons = ButtonOutlierWindow()
        # Now connect buttons to functions
        self.outlier_buttons.button_vec_field_r.clicked.connect(self.button_vec_field_r_clicked)
        self.outlier_buttons.button_vec_field_c.clicked.connect(self.button_vec_field_c_clicked)
        self.outlier_buttons.button_shift_cmap_r.clicked.connect(self.button_shift_cmap_r_clicked)
        self.outlier_buttons.button_shift_cmap_c.clicked.connect(self.button_shift_cmap_c_clicked)
        self.outlier_buttons.button_scale_r.clicked.connect(self.button_scale_r_clicked)
        self.outlier_buttons.button_scale_c.clicked.connect(self.button_scale_c_clicked)

        # Finally, add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.outlier_buttons, area="left", name='Cross Tile Outlier Removal',
                                           add_vertical_stretch=False)

        # Create a single widget containing buttons for ICP diagnostics
        self.icp_buttons = ButtonICPWindow()
        # Now connect buttons to functions
        self.icp_buttons.button_mse.clicked.connect(self.button_mse_clicked)
        self.icp_buttons.button_matches.clicked.connect(self.button_matches_clicked)
        self.icp_buttons.button_deviations.clicked.connect(self.button_deviations_clicked)

        # Finally, add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.icp_buttons, area="left", name='ICP Diagnostics',
                                                add_vertical_stretch=False)

        # Get target images and anchor image
        self.get_images()

        # Plot images
        self.plot_images()

        napari.run()

    # Button functions

    # Tiles grid
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

    # Contrast
    def change_anchor_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(0, 32, 2):
            self.viewer.layers[i].contrast_limits = [low, high]

    # contrast
    def change_imaging_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(1, 32, 2):
            self.viewer.layers[i].contrast_limits = [low, high]

    # method
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

    # method
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

    # SVR
    def button_pearson_hist_clicked(self):
        self.svr_buttons.pearson_hist.setChecked(True)
        # link this to the function that plots the histogram of correlation coefficients
        view_pearson_hists(nb=self.nb, t=self.tile)

    # SVR
    def button_pearson_cmap_clicked(self):
        self.svr_buttons.pearson_cmap.setChecked(True)
        # link this to the function that plots the histogram of correlation coefficients
        view_pearson_colourmap(nb=self.nb, t=self.tile)

    # SVR
    def button_pearson_spatial_round_clicked(self):
        self.svr_buttons.pearson_spatial_round.setChecked(True)
        view_pearson_colourmap_spatial(nb=self.nb, t=self.tile, round=True)

    # SVR
    def button_pearson_spatial_channel_clicked(self):
        self.svr_buttons.pearson_spatial_channel.setChecked(True)
        view_pearson_colourmap_spatial(nb=self.nb, t=self.tile, round=False)

    # SVR
    def create_round_slot(self, r):

        def round_button_clicked():
            use_rounds = self.nb.basic_info.use_rounds
            for rnd in use_rounds:
                self.svr_buttons.__getattribute__('R'+str(rnd)).setChecked(rnd == r)
            # We don't need to update the plot, we just need to call the viewing function
            view_regression_scatter(nb=self.nb, t=self.tile, index=r, round=True)
        return round_button_clicked

    # SVR
    def create_channel_slot(self, c):

        def channel_button_clicked():
            use_channels = self.nb.basic_info.use_channels
            for chan in use_channels:
                self.svr_buttons.__getattribute__('C'+str(chan)).setChecked(chan == c)
            # We don't need to update the plot, we just need to call the viewing function
            view_regression_scatter(nb=self.nb, t=self.tile, index=c, round=False)
        return channel_button_clicked

    # outlier removal
    def button_vec_field_r_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_vec_field_r.setChecked(True)
        shift_vector_field(nb=self.nb, round=True)

    # outlier removal
    def button_vec_field_c_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_vec_field_c.setChecked(True)
        shift_vector_field(nb=self.nb, round=False)

    # outlier removal
    def button_shift_cmap_r_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_shift_cmap_r.setChecked(True)
        zyx_shift_image(nb=self.nb, round=True)

    # outlier removal
    def button_shift_cmap_c_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_shift_cmap_c.setChecked(True)
        zyx_shift_image(nb=self.nb, round=False)

    # outlier removal
    def button_scale_r_clicked(self):
        # No need to only allow one button pressed
        self.outlier_buttons.button_scale_r.setChecked(True)
        view_round_scales(nb=self.nb)

    # outlier removal
    def button_scale_c_clicked(self):
        view_channel_scales(nb=self.nb)

    # icp
    def button_mse_clicked(self):
        view_icp_mse(nb=self.nb, t=self.tile)

    # icp
    def button_matches_clicked(self):
        view_icp_n_matches(nb=self.nb, t=self.tile)

    # icp
    def button_deviations_clicked(self):
        view_icp_deviations(nb=self.nb, t=self.tile)

    # Button functions end here
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
        features = {'r': np.repeat(np.append(use_rounds, np.ones(len(use_channels)) * self.r_mid), 10).astype(int),
                            'c': np.repeat(np.append(np.ones(len(use_rounds)) * self.c_ref, use_channels), 10).astype(int)}
        features_anchor = {'r': np.repeat(np.ones(len(use_rounds) + len(use_channels)) * self.r_ref, 10).astype(int),
                            'c': np.repeat(np.ones(len(use_rounds) + len(use_channels)) * self.c_ref, 10).astype(int)}

        # Define text
        text = {
            'string': 'R: {r} C: {c}',
            'size': 8,
            'color': 'green'}
        text_anchor = {
            'string': 'R: {r} C: {c}',
            'size': 8,
            'color': 'red'}

        # Now go on to define point coords. Napari only allows us to plot text with points, so will plot points that
        # are not visible and attach text to them
        points = []

        for r in use_rounds:
            self.viewer.add_image(self.base_image, blending='additive', colormap='red', translate=[0, 0, 1_000 * r],
                                  name='Anchor', contrast_limits=[0, 100])
            self.viewer.add_image(self.target_round_image[r], blending='additive', colormap='green',
                                  translate=[0, 0, 1_000 * r], name='Round ' + str(r) + ', Channel ' + str(self.c_ref),
                                  contrast_limits=[0, 100])
            # Add this to all z planes so still shows up when scrolling
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

        points = np.array(points)
        # Add text to image
        self.viewer.add_points(points, features=features, text=text, size=1)
        self.viewer.add_points(points - [0, 100, 0], features=features_anchor, text=text_anchor, size=1)

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


class ButtonSVRWindow(QMainWindow):
    # This class creates a window with buttons for all SVR diagnostics
    # This includes buttons for each round and channel regression
    # Also includes a button to view pearson correlation coefficient in either a histogram or colormap
    # Also includes a button to view pearson correlation coefficient spatially for either rounds or channels
    def __init__(self, use_rounds: list, use_channels: list):
        super().__init__()
        # Create round regression buttons
        for r in use_rounds:
            # Create a button for each tile
            button = QPushButton('R' + str(r), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(True)
            x, y_r = r % 4, r // 4
            button.setGeometry(x * 70, 40 + 60 * y_r, 50, 28)
            # Finally add this button as an attribute to self
            self.__setattr__('R' + str(r), button)

        # create channel regression buttons
        for c in range(len(use_channels)):
            # Create a button for each tile
            button = QPushButton('C' + str(use_channels[c]), self)
            # set the button to be checkable iff t in use_tiles
            button.setCheckable(True)
            x, y_c = c % 4, y_r + c // 4 + 1
            button.setGeometry(x * 70, 40 + 60 * y_c, 50, 28)
            # Finally add this button as an attribute to self
            self.__setattr__('C' + str(use_channels[c]), button)

        # Create 2 correlation buttons:
        # 1 to view pearson correlation coefficient as histogram
        # 2 to view pearson correlation coefficient as colormap
        y = y_c + 1
        button = QPushButton('r_hist', self)
        button.setCheckable(True)
        button.setGeometry(0, 40 + 60 * y, 120, 28)
        self.pearson_hist = button
        button = QPushButton('r_cmap', self)
        button.setCheckable(True)
        button.setGeometry(140, 40 + 60 * y, 120, 28)
        self.pearson_cmap = button

        # Create 2 spatial correlation buttons:
        # 1 to view pearson correlation coefficient spatially for rounds
        # 2 to view pearson correlation coefficient spatially for channels
        y += 1
        button = QPushButton('r_spatial_round', self)
        button.setCheckable(True)
        button.setGeometry(0, 40 + 60 * y, 120, 28)
        self.pearson_spatial_round = button
        button = QPushButton('r_spatial_channel', self)
        button.setCheckable(True)
        button.setGeometry(140, 40 + 60 * y, 120, 28)
        self.pearson_spatial_channel = button


class ButtonOutlierWindow(QMainWindow):
    # This class creates a window with buttons for all outlier removal diagnostics
    # This includes round and channel button to view shifts for each tile as a vector field
    # Also includes round and channel button to view shifts for each tile as a heatmap
    # Also includes round and channel button to view boxplots of scales for each tile
    def __init__(self):
        super().__init__()
        self.button_vec_field_r = QPushButton('Round Shift Vector Field', self)
        self.button_vec_field_r.setCheckable(True)
        self.button_vec_field_r.setGeometry(20, 40, 220, 28)

        self.button_vec_field_c = QPushButton('Channel Shift Vector Field', self)
        self.button_vec_field_c.setCheckable(True)
        self.button_vec_field_c.setGeometry(20, 100, 220, 28)  # left, top, width, height

        self.button_shift_cmap_r = QPushButton('Round Shift Colour Map', self)
        self.button_shift_cmap_r.setCheckable(True)
        self.button_shift_cmap_r.setGeometry(20, 160, 220, 28)

        self.button_shift_cmap_c = QPushButton('Channel Shift Colour Map', self)
        self.button_shift_cmap_c.setCheckable(True)
        self.button_shift_cmap_c.setGeometry(20, 220, 220, 28)  # left, top, width, height

        self.button_scale_r = QPushButton('Round Scales', self)
        self.button_scale_r.setCheckable(True)
        self.button_scale_r.setGeometry(20, 280, 100, 28)

        self.button_scale_c = QPushButton('Channel Scales', self)
        self.button_scale_c.setCheckable(True)
        self.button_scale_c.setGeometry(140, 280, 100, 28)  # left, top, width, height


class ButtonICPWindow(QMainWindow):
    # This class creates a window with buttons for all ICP diagnostics
    # One diagnostic for MSE, one for n_matches, one for icp_deciations
    def __init__(self):
        super().__init__()
        self.button_mse = QPushButton('MSE', self)
        self.button_mse.setCheckable(True)
        self.button_mse.setGeometry(20, 40, 100, 28)

        self.button_matches = QPushButton('Matches', self)
        self.button_matches.setCheckable(True)
        self.button_matches.setGeometry(140, 40, 100, 28)  # left, top, width, height

        self.button_deviations = QPushButton('Large ICP Deviations', self)
        self.button_deviations.setCheckable(True)
        self.button_deviations.setGeometry(20, 100, 220, 28)


def set_style(button):
    # Set button color = grey when hovering over, blue when pressed, white when not
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
    return button


# 1
def view_regression_scatter(nb: Notebook, t: int, index: int, round: bool = True):
    """
    view 3 scatter plots for each data set shift vs positions
    Args:
        nb: Notebook
        t: tile
        index: round index if round, else channel index
        round: True if round, False if channel
    """
    # Transpose shift and position variables so coord is dimension 0, makes plotting easier
    if round:
        mode = 'Round'
        shift = nb.register_debug.round_shift[t, index].T
        subvol_transform = nb.register.round_transform[t, index]
        icp_transform = yxz_to_zyx_affine(A=nb.register.transform[t, index, nb.basic_info.anchor_channel],
                                          z_scale= nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy)
    else:
        mode = 'Channel'
        shift = nb.register_debug.channel_shift[t, index].T
        subvol_transform = nb.register.channel_transform[t, index]
        A = yxz_to_zyx_affine(A=nb.register.transform[t, nb.basic_info.n_rounds // 2, index],
                                       z_scale= nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy)
        B = yxz_to_zyx_affine(A=nb.register.transform[t, nb.basic_info.n_rounds // 2, nb.basic_info.anchor_channel],
                                        z_scale= nb.basic_info.pixel_size_z / nb.basic_info.pixel_size_xy)
        icp_transform = compose_affine(A, invert_affine(B))
    position = nb.register_debug.position.T

    # Make ranges, wil be useful for plotting lines
    z_range = np.arange(np.min(position[0]), np.max(position[0]))
    yx_range = np.arange(np.min(position[1]), np.max(position[1]))
    coord_range = [z_range, yx_range, yx_range]
    # Need to add a central offset to all lines plotted
    tile_centre_zyx = np.roll(nb.basic_info.tile_centre, 1)

    # We want to plot the shift of each coord against the position of each coord. The gradient when the dependent var
    # is coord i and the independent var is coord j should be the transform[i,j] - int(i==j)
    gradient_svr = subvol_transform[:3, :3] - np.eye(3)
    gradient_icp = icp_transform[:3, :3] - np.eye(3)
    # Now we need to compute what the intercept should be for each coord. Usually this would just be given by the final
    # column of the transform, but we need to add a central offset to this. If the dependent var is coord i, and the
    # independent var is coord j, then the intercept should be the transform[i,3] + central_offset[i,j]. This central
    # offset is given by the formula: central_offset[i, j] = gradient[i, k1] * tile_centre[k1] + gradient[i, k2] *
    # tile_centre[k2], where k1 and k2 are the coords that are not j.
    central_offset_svr = np.zeros((3, 3))
    central_offset_icp = np.zeros((3, 3))
    intercpet_svr = np.zeros((3, 3))
    intercpet_icp = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # k1 and k2 are the coords that are not j
            k1 = (j + 1) % 3
            k2 = (j + 2) % 3
            central_offset_svr[i, j] = gradient_svr[i, k1] * tile_centre_zyx[k1] + gradient_svr[i, k2] * tile_centre_zyx[k2]
            central_offset_icp[i, j] = gradient_icp[i, k1] * tile_centre_zyx[k1] + gradient_icp[i, k2] * tile_centre_zyx[k2]
            # Now compute the intercepts
            intercpet_svr[i, j] = subvol_transform[i, 3] + central_offset_svr[i, j]
            intercpet_icp[i, j] = icp_transform[i, 3] + central_offset_icp[i, j]

    # Define the axes
    fig, axes = plt.subplots(3, 3)
    coord = ['Z', 'Y', 'X']
    # Now plot n_matches
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.scatter(x=position[j], y=shift[i], alpha=0.3)
            ax.plot(coord_range[j], gradient_svr[i, j] * coord_range[j] + intercpet_svr[i, j], label='SVR')
            ax.plot(coord_range[j], gradient_icp[i, j] * coord_range[j] + intercpet_icp[i, j], label='ICP')
            ax.legend()
    # Label subplot rows and columns with coord names
    for ax, col in zip(axes[0], coord):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], coord):
        ax.set_ylabel(row, rotation=90, size='large')
    # common axis labels
    fig.supxlabel('Position')
    fig.supylabel('Shift')
    # Add title
    plt.suptitle(mode + ' regression for Tile ' + str(t) + ', ' + mode + ' ' + str(index))
    plt.show()


# 1
def view_pearson_hists(nb, t, num_bins=30):
    """
    function to view histogram of correlation coefficients for all subvol shifts of all round/channels.
    Args:
        nb: Notebook
        t: int tile under consideration
        num_bins: int number of bins in the histogram
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    thresh = nb.get_config()['register']['r_thresh']
    round_corr, channel_corr = nbp_register_debug.round_shift_corr[t], nbp_register_debug.channel_shift_corr[t]
    n_rounds, n_channels_use = nbp_basic.n_rounds, len(nbp_basic.use_channels)
    use_channels = nbp_basic.use_channels
    cols = max(n_rounds, n_channels_use)

    for r in range(n_rounds):
        plt.subplot(2, cols, r + 1)
        counts, _ = np.histogram(round_corr[r], np.linspace(0, 1, num_bins))
        plt.hist(round_corr[r], bins=np.linspace(0, 1, num_bins))
        plt.vlines(x=thresh, ymin=0, ymax=np.max(counts), colors='r')
        # change fontsize from default 10 to 7
        plt.title('r = ' + str(r) +
                  '\n Pass = ' + str(
            round(100 * sum(round_corr[r] > thresh) / round_corr.shape[1], 2)) + '%', fontsize=7)
        # remove x ticks and y ticks
        plt.xticks([])
        plt.yticks([])

    for c in range(n_channels_use):
        plt.subplot(2, cols, cols + c + 1)
        counts, _ = np.histogram(channel_corr[use_channels[c]], np.linspace(0, 1, num_bins))
        plt.hist(channel_corr[use_channels[c]], bins=np.linspace(0, 1, num_bins))
        if c == 0:
            plt.vlines(x=thresh, ymin=0, ymax=np.max(counts), colors='r', label='r_thresh = ' + str(thresh))
        else:
            plt.vlines(x=thresh, ymin=0, ymax=np.max(counts), colors='r')
        # change fontsize from default 10 to 7
        plt.title('c = ' + str(use_channels[c]) +
                  '\n Pass = ' + str(
            round(100 * sum(channel_corr[use_channels[c]] > thresh) / channel_corr.shape[1], 2)) + '%', fontsize=7)
        # remove x ticks and y ticks
        plt.xticks([])
        plt.yticks([])

    plt.suptitle('Similarity Score Distributions for all Sub-Volume Shifts')
    plt.show()


# 1
def view_pearson_colourmap(nb, t):
    """
    function to view colourmap of correlation coefficients for all subvol shifts for all channels and rounds.

    Args:
        nb: Notebook
        t: int tile under consideration
    """
    # initialise frequently used variables
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    round_corr, channel_corr = nbp_register_debug.round_shift_corr[t], \
        nbp_register_debug.channel_shift_corr[t, nbp_basic.use_channels]
    use_channels = nbp_basic.use_channels
    # Replace 0 with nans so they get plotted as black
    round_corr[round_corr == 0] = np.nan
    channel_corr[channel_corr == 0] = np.nan

    # plot round correlation and tile correlation
    fig, axes = plt.subplots(2, 1)
    ax1, ax2 = axes[0], axes[1]
    # ax1 refers to round shifts
    im = ax1.imshow(round_corr, vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax1.set_xlabel('Sub-volume index')
    ax1.set_ylabel('Round')
    ax1.set_title('Round sub-volume shift scores')
    # ax2 refers to channel shifts
    im = ax2.imshow(channel_corr, vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax2.set_xlabel('Sub-volume index')
    ax2.set_ylabel('Channel')
    ax2.set_yticks(np.arange(len(nbp_basic.use_channels)), nbp_basic.use_channels)
    ax2.set_title('Channel sub-volume shift scores')

    # Add common colour bar. Also give it the label 'Pearson correlation coefficient'
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Correlation coefficient')

    plt.suptitle('Similarity score distributions for all sub-volume shifts')


# 1
def view_pearson_colourmap_spatial(nb: Notebook, t: int, round: bool = True):
    """
    function to view colourmap of correlation coefficients along with spatial info for either all round shifts of a tile
    or all channel shifts of a tile.

    Args:
        nb: Notebook
        round: True if round, false if channel
        t: tile under consideration
    """

    # initialise frequently used variables
    config = nb.get_config()['register']
    if round:
        use = nb.basic_info.use_rounds
        corr = nb.register_debug.round_shift_corr[t, use]
        mode = 'Round'
    else:
        use = nb.basic_info.use_channels
        corr = nb.register_debug.channel_shift_corr[t, use]
        mode = 'Channel'

    # Set 0 correlations to nan, so they are plotted as black
    corr[corr == 0] = np.nan
    z_subvols, y_subvols, x_subvols = config['z_subvols'], config['y_subvols'], config['x_subvols']
    n_rc = corr.shape[0]

    fig, axes = plt.subplots(nrows=z_subvols, ncols=n_rc)
    # Now plot each image
    for elem in range(n_rc):
        for z in range(z_subvols):
            ax = axes[z, elem]
            ax.set_xticks([])
            ax.set_yticks([])
            im = ax.imshow(np.reshape(corr[elem, z * y_subvols * x_subvols: (z + 1) * x_subvols * y_subvols],
                                      (y_subvols, x_subvols)), vmin=0, vmax=1)
    # common axis labels
    fig.supxlabel(mode)
    fig.supylabel('Z-Subvolume')
    # Set row and column labels
    for ax, col in zip(axes[0], use):
        ax.set_title(col, size='large')
    for ax, row in zip(axes[:, 0], np.arange(z_subvols)):
        ax.set_ylabel(row, rotation=0, size='large', x=-0.1)
    # add colour bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Correlation coefficient')

    plt.suptitle(mode + ' shift similarity scores for tile ' + str(t) + ' plotted spatially')


# 2
def shift_vector_field(nb: Notebook, round: bool = True):
    """
    Function to plot vector fields of predicted shifts vs shifts to see if we classify a shift as an outlier.
    Args:
        nb: Notebook
        round: True if round, False if Channel
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    residual_thresh = nb.get_config()['register']['residual_thresh']
    use_tiles = nbp_basic.use_tiles
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]

    # Load in shift
    if round:
        mode = 'Round'
        use_rc = nbp_basic.use_rounds
        shift = nbp_register_debug.round_transform_unregularised[use_tiles, :, :, 3][:, use_rc]
    else:
        mode = 'Channel'
        use_rc = nbp_basic.use_channels
        shift = nbp_register_debug.channel_transform_unregularised[use_tiles, :, :, 3][:, use_rc]

    # record number of rounds/channels, tiles and initialise predicted shift
    n_t, n_rc = shift.shape[0], len(use_rc)
    tilepos_yx_pad = np.vstack((tilepos_yx.T, np.ones(n_t))).T
    predicted_shift = np.zeros_like(shift)

    fig, axes = plt.subplots(nrows=3, ncols=n_rc)
    for elem in range(n_rc):
        # generate predicted shift for this r/c via huber regression
        transform = huber_regression(shift[:, elem], tilepos_yx)
        predicted_shift[:, elem] = tilepos_yx_pad @ transform.T
        scale = 2 * np.sqrt(np.sum(predicted_shift[:, elem, 1:] ** 2, axis=1))

        # plot the predicted yx shift vs actual yx shift in row 0
        ax = axes[0, elem]
        # Make sure the vector field is properly scaled
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], predicted_shift[:, elem, 2], predicted_shift[:, elem, 1],
                  color='b', scale=scale, scale_units='width', width=.05, alpha=0.5)
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], shift[:, elem, 2], shift[:, elem, 1], color='r', scale=scale,
                  scale_units='width', width=.05, alpha=0.5)
        # We want to set the xlims and ylims to include a bit of padding so we can see the vectors
        ax.set_xlim(tilepos_yx[:, 1].min() - 1, tilepos_yx[:, 1].max() + 1)
        ax.set_ylim(tilepos_yx[:, 0].min() - 1, tilepos_yx[:, 0].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('XY shifts')

        # plot the predicted z shift vs actual z shift in row 1
        ax = axes[1, elem]
        # we only want 1 label so make this for elem = 0
        if elem == 0:
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, predicted_shift[:, elem, 0], color='b',
                      label='regularised', scale=scale, scale_units='width', width=.05, alpha=0.5)
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, shift[:, elem, 2], color='r', label='raw', scale=scale,
                      scale_units='width', width=.05, alpha=0.5)
            # We want to set the xlims and ylims to include a bit of padding so we can see the vectors
            ax.set_xlim(tilepos_yx[:, 1].min() - 1, tilepos_yx[:, 1].max() + 1)
            ax.set_ylim(tilepos_yx[:, 0].min() - 1, tilepos_yx[:, 0].max() + 1)
        else:
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, predicted_shift[:, elem, 0], color='b', scale=scale,
                      scale_units='width', width=.05, alpha=0.5)
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, shift[:, elem, 2], color='r', scale=scale,
                      scale_units='width', width=.05, alpha=0.5)
            # We want to set the xlims and ylims to include a bit of padding so we can see the vectors
            ax.set_xlim(tilepos_yx[:, 1].min() - 1, tilepos_yx[:, 1].max() + 1)
            ax.set_ylim(tilepos_yx[:, 0].min() - 1, tilepos_yx[:, 0].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Z shifts')

        # Plot image of norms of residuals at each tile in row 3
        ax = axes[2, elem]
        diff = create_tiled_image(data=np.linalg.norm(predicted_shift[:, elem] - shift[:, elem], axis=1),
                                  nbp_basic=nbp_basic)
        outlier = np.argwhere(diff > residual_thresh)
        n_outliers = outlier.shape[0]
        im = ax.imshow(diff, vmin=0, vmax=10)
        # Now we want to outline the outlier pixels with a dotted red rectangle
        for i in range(n_outliers):
            rect = patches.Rectangle((outlier[i, 1] - 0.5, outlier[i, 0] - 0.5), 1, 1, linewidth=1, edgecolor='r',
                                     facecolor='none', linestyle='--')
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Residual')

    # Set row and column labels
    for ax, col in zip(axes[0], use_rc):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], ['XY-shifts', 'Z-shifts', 'Residuals']):
        ax.set_ylabel(row, rotation=90, size='large')
    # Add row and column labels
    fig.supxlabel(mode)
    fig.supylabel('Diagnostic')
    # Add global colour bar and legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Residual Norm')
    # Add title
    fig.suptitle('Diagnostic plots for {} shift outlier removal'.format(mode), size='x-large')


# 2
def zyx_shift_image(nb: Notebook, round: bool = True):
    """
        Function to plot overlaid images of predicted shifts vs shifts to see if we classify a shift as an outlier.
        Args:
            nb: Notebook
            round: Boolean indicating whether we are looking at round outlier removal, True if r, False if c
    """
    nbp_basic, nbp_register, nbp_register_debug = nb.basic_info, nb.register, nb.register_debug
    use_tiles = nbp_basic.use_tiles

    # Load in shift
    if round:
        mode = 'Round'
        use = nbp_basic.use_rounds
        shift_raw = nbp_register_debug.round_transform_unregularised[use_tiles, :, :, 3]
        shift = nbp_register.round_transform[use_tiles, :, :, 3]
    else:
        mode = 'Channel'
        use = nbp_basic.use_channels
        shift_raw = nbp_register_debug.channel_transform_unregularised[use_tiles, :, :, 3]
        shift = nbp_register.channel_transform[use_tiles, :, :, 3]

    coord_label = ['Z', 'Y', 'X']
    n_t, n_rc = shift.shape[0], shift.shape[1]
    fig, axes = plt.subplots(nrows=3, ncols=n_rc)
    # common axis labels
    fig.supxlabel(mode)
    fig.supylabel('Coordinate (Z, Y, X)')

    # Set row and column labels
    for ax, col in zip(axes[0], use):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], coord_label):
        ax.set_ylabel(row, rotation=0, size='large')

    # Now we will plot 3 rows of subplots and n_rc columns of subplots. Each subplot will be made up of 2 further subplots
    # The Left subplot will be the raw shift and the right will be the regularised shift
    # We will also outline pixels in these images that are different between raw and regularised with a dotted red rectangle
    for elem in range(n_rc):
        for coord in range(3):
            ax = axes[coord, elem]
            # remove the ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Create 2 subplots within each subplot
            gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=ax, wspace=0.1)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            # Plot the raw shift in the left subplot
            im = ax1.imshow(create_tiled_image(shift_raw[:, elem, coord], nbp_basic))
            # Plot the regularised shift in the right subplot
            im = ax2.imshow(create_tiled_image(shift[:, elem, coord], nbp_basic))
            # Now we want to outline the pixels that are different between raw and regularised with a dotted red rectangle
            diff = np.abs(shift_raw[:, elem, coord] - shift[:, elem, coord])
            outlier = np.argwhere(diff > 0.1)
            n_outliers = outlier.shape[0]
            for i in range(n_outliers):
                rect = patches.Rectangle((outlier[i, 1] - 0.5, outlier[i, 0] - 0.5), 1, 1, linewidth=1, edgecolor='r',
                                         facecolor='none', linestyle='--')
                # Add the rectangle to both subplots
                ax1.add_patch(rect)
                ax2.add_patch(rect)
                # Remove ticks and labels from the left subplot
                ax1.set_xticks([])
                ax1.set_yticks([])
                # Remove ticks and labels from the right subplot
                ax2.set_xticks([])
                ax2.set_yticks([])

    fig.canvas.draw()
    plt.show()
    # Add a title
    fig.suptitle('Diagnostic plots for {} shift outlier removal'.format(mode), size='x-large')
    # add a global colour bar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


# 2
def create_tiled_image(data, nbp_basic):
    """
    generate image of 1d tile data along with tile positions
    Args:
        data: n_tiles_use x 1 list of residuals
        nbp_basic: basic info notebook page
    """
    # Initialise frequently used variables
    use_tiles = nbp_basic.use_tiles
    tilepos_yx = nbp_basic.tilepos_yx[nbp_basic.use_tiles]
    n_rows = np.max(tilepos_yx[:, 0]) - np.min(tilepos_yx[:, 0]) + 1
    n_cols = np.max(tilepos_yx[:, 1]) - np.min(tilepos_yx[:, 1]) + 1
    tilepos_yx = tilepos_yx - np.min(tilepos_yx, axis=0)
    diff = np.zeros((n_rows, n_cols))

    for t in range(len(use_tiles)):
        diff[tilepos_yx[t, 1], tilepos_yx[t, 0]] = data[t]

    diff = np.flip(diff.T, axis=-1)
    diff[diff == 0] = np.nan

    return diff


# 2
def view_round_scales(nb: Notebook):
    """
    view scale parameters for the round outlier removals
    Args:
        nb: Notebook
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    anchor_round, anchor_channel = nbp_basic.anchor_round, nbp_basic.anchor_channel
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
    plt.plot(np.arange(n_rounds), np.percentile(z_scale, 25, axis=0), 'c:', label='Inter Quartile Range')
    plt.plot(np.arange(n_rounds), np.percentile(z_scale, 50, axis=0), 'r:', label='Median')
    plt.plot(np.arange(n_rounds), np.percentile(z_scale, 75, axis=0), 'c:')
    plt.xlabel('Rounds')
    plt.ylabel('Z-scales')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.scatter(x=np.tile(np.arange(n_rounds), n_tiles_use),
                y=np.reshape(y_scale, (n_tiles_use * n_rounds)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_rounds), 0.999 * np.ones(n_rounds), 'c:', label='0.999 - 1.001')
    plt.plot(np.arange(n_rounds), np.ones(n_rounds), 'r:', label='1')
    plt.plot(np.arange(n_rounds), 1.001 * np.ones(n_rounds), 'c:')
    plt.xlabel('Rounds')
    plt.ylabel('Y-scales')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.scatter(x=np.tile(np.arange(n_rounds), n_tiles_use),
                y=np.reshape(x_scale, (n_tiles_use * n_rounds)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_rounds), 0.999 * np.ones(n_rounds), 'c:', label='0.999 - 1.001')
    plt.plot(np.arange(n_rounds), np.ones(n_rounds), 'r:', label='1')
    plt.plot(np.arange(n_rounds), 1.001 * np.ones(n_rounds), 'c:')
    plt.xlabel('Rounds')
    plt.ylabel('X-scales')
    plt.legend()

    plt.suptitle('Distribution of scales across tiles for registration from Anchor (R: ' + str(anchor_round)
                 + ', C: ' + str(anchor_channel) + ') to the reference channel of imaging rounds (R: r, C: '
                 + str(anchor_channel) + ') for all rounds r.')
    plt.show()


# 2
def view_channel_scales(nb: Notebook):
    """
    view scale parameters for the round outlier removals
    Args:
        nb: Notebook
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    mid_round, anchor_channel = nbp_basic.n_rounds // 2, nbp_basic.anchor_channel
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
    plt.plot(np.arange(n_channels_use), 0.99 * np.ones(n_channels_use), 'c:', label='0.99 - 1.01')
    plt.plot(np.arange(n_channels_use), np.ones(n_channels_use), 'r:', label='1')
    plt.plot(np.arange(n_channels_use), 1.01 * np.ones(n_channels_use), 'c:')
    plt.xticks(np.arange(n_channels_use), use_channels)
    plt.xlabel('Channel')
    plt.ylabel('Z-scale')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.scatter(x=np.tile(np.arange(n_channels_use), n_tiles_use),
                y=np.reshape(y_scale, (n_tiles_use * n_channels_use)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 25, axis=0), 'c:', label='Inter Quartile Range')
    plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 50, axis=0), 'r:', label='Median')
    plt.plot(np.arange(n_channels_use), np.percentile(y_scale, 75, axis=0), 'c:')
    plt.xticks(np.arange(n_channels_use), use_channels)
    plt.xlabel('Channel')
    plt.ylabel('Y-scale')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.scatter(x=np.tile(np.arange(n_channels_use), n_tiles_use),
                y=np.reshape(x_scale, (n_tiles_use * n_channels_use)), c='w', marker='x', alpha=0.7)
    plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 25, axis=0), 'c:', label='Inter Quartile Range')
    plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 50, axis=0), 'r:', label='Median')
    plt.plot(np.arange(n_channels_use), np.percentile(x_scale, 75, axis=0), 'c:')
    plt.xticks(np.arange(n_channels_use), use_channels)
    plt.xlabel('Channel')
    plt.ylabel('X-scale')
    plt.legend()

    plt.suptitle('Distribution of scales across tiles for registration from adjusted anchor in coordinate frame of (R: '
                 + str(mid_round) + ', C: ' + str(anchor_channel) + ') to (R:' + str(mid_round) + ' C: c for all '
                                                                                                  'channels c.')
    plt.show()


# 3
def view_icp_n_matches(nb: Notebook, t: int):
    """
    Plots simple proportion matches against iterations.
    Args:
        nb: Notebook
        t: tile
    """
    nbp_basic, nbp_register_debug, nbp_find_spots = nb.basic_info, nb.register_debug, nb.find_spots
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    n_matches = nbp_register_debug.n_matches[t, use_rounds][:, use_channels]
    frac_matches = n_matches_to_frac_matches(n_matches=n_matches,
                                             spot_no=nbp_find_spots.spot_no[t, use_rounds][:, use_channels])
    n_iters = n_matches.shape[2]

    # Define the axes
    fig, axes = plt.subplots(len(use_rounds), len(use_channels))
    # common axis labels
    fig.supxlabel('Channels')
    fig.supylabel('Rounds')
    # Set row and column labels
    for ax, col in zip(axes[0], use_channels):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], use_rounds):
        ax.set_ylabel(row, rotation=0, size='large')

    # Now plot n_matches
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            ax = axes[r, c]
            ax.plot(np.arange(n_iters), frac_matches[r, c])
            ax.set_xticks([])
            ax.set_yticks([0, 1])

    plt.suptitle('Fraction of imaging spots matched against iteration of ICP for tile ' + str(t) +
                 ' for all rounds and channels, for all ' + str(n_iters) + ' iterations.')
    plt.show()


# 3
def view_icp_mse(nb: Notebook, t: int):
    """
    Plots simple MSE grid against iterations
    Args:
        nb: Notebook
        t: tile
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    mse = nbp_register_debug.mse[t, use_rounds][:, use_channels]
    n_iters = mse.shape[2]

    # Define the axes
    fig, axes = plt.subplots(len(use_rounds), len(use_channels))
    # common axis labels
    fig.supxlabel('Channels')
    fig.supylabel('Rounds')
    # Set row and column labels
    for ax, col in zip(axes[0], use_channels):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], use_rounds):
        ax.set_ylabel(row, rotation=0, size='large')

    # Now plot mse
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            ax = axes[r, c]
            ax.plot(np.arange(n_iters), mse[r, c])
            ax.set_xticks([])
            ax.set_yticks([np.min(mse[r, c]), np.max(mse[r, c])])

    plt.suptitle('MSE against iteration of ICP for tile ' + str(t) + ' for all rounds and channels, for all '
                 + str(n_iters) + ' iterations.')
    plt.show()


# 3
def view_icp_deviations(nb: Notebook, t: int):
    """
    Plots deviations of ICP transform for a given tile t (n_rounds x n_channel x 3 x 4) affine transform against initial
    guess (subvol_transform) which has the same shape. These trasnforms are in zyx x zyx format, with the final col
    referring to the shift. Our plot has rows as rounds and columns as channels, giving us len(use_rounds) rows, and
    len(use_channels) columns of subplots.

    Each subplot will be a 2 3x1 images where the first im is [z_scale_icp - z_scale_svr, y_scale_icp - y_scale_svr,
    x_scale_icp - x_scale_svr], and second im is [z_shift_icp - z_shift_svr, y_shift_icp - y_shift_svr,
    s_shift_icp - x_shift_svr]. There should be a common colour bar on the right for all scale difference images and
    another on the right for all shift difference images.

    Args:
        nb: Notebook
        t: tile
    """
    # Initialise frequent variables
    nbp_basic, nbp_register = nb.basic_info, nb.register
    use_tiles, use_rounds, use_channels = nbp_basic.use_tiles, nbp_basic.use_rounds, nbp_basic.use_channels
    subvol_transform = np.zeros((len(use_rounds), len(use_channels), 3, 4))
    transform = np.zeros((len(use_rounds), len(use_channels), 3, 4))
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            subvol_transform[r, c] = yxz_to_zyx_affine(A=nbp_register.subvol_transform[t, use_rounds[r], use_channels[c]],
                                                         z_scale=nbp_basic.pixel_size_z/nbp_basic.pixel_size_xy)
            transform[r, c] = yxz_to_zyx_affine(A=nbp_register.transform[t, use_rounds[r], use_channels[c]],
                                                         z_scale=nbp_basic.pixel_size_z/nbp_basic.pixel_size_xy)

    # Define the axes
    fig, axes = plt.subplots(len(use_rounds), len(use_channels))
    # common axis labels
    fig.supxlabel('Channels')
    fig.supylabel('Rounds')
    # Set row and column labels
    for ax, col in zip(axes[0], use_channels):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], use_rounds):
        ax.set_ylabel(row, rotation=0, size='large')

    # Define difference images
    scale_diff = np.zeros((len(use_rounds), len(use_channels), 3))
    shift_diff = np.zeros((len(use_rounds), len(use_channels), 3))
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            scale_diff[r, c] = np.diag(transform[r, c, :3, :3]) - np.diag(subvol_transform[r, c, :3, :3])
            shift_diff[r, c] = transform[r, c, :3, 3] - subvol_transform[r, c, :3, 3]
    
    # Now plot scale_diff
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            ax = axes[r, c]
            # create 2 subplots within this subplot
            ax1 = ax.inset_axes([0, 0, 0.5, 1])
            ax2 = ax.inset_axes([0.5, 0, 0.5, 1])
            # plot scale_diff
            im1 = ax1.imshow(scale_diff[r, c].reshape(3, 1), cmap='bwr', vmin=-.1, vmax=.1)
            # plot shift_diff
            im2 = ax2.imshow(shift_diff[r, c].reshape(3, 1), cmap='bwr', vmin=-5, vmax=5)
            # remove ticks
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax.set_xticks([])
            ax.set_yticks([])

    # plot 2 colour bars, one for the shift_diff and one for the scale_diff. Both colour bars should be the same size,
    # and the scale_diff colour bar should be on the left of the subplots, and the shift_diff colour bar should be on
    # the right of the subplots.
    fig.subplots_adjust(right=0.8)
    cbar_scale_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
    # Next we want to make sure the scale cbar has ticks on the left.
    cbar_scale_ax.yaxis.tick_left()
    fig.colorbar(im1, cax=cbar_scale_ax, ticks=[-.1, 0, .1])
    cbar_shift_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    fig.colorbar(im2, cax=cbar_shift_ax, ticks=[-5, 0, 5])

    plt.suptitle('Deviations of ICP transform against initial guess for tile ' + str(t) + ' for all rounds and '
                    'channels.')
