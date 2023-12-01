import os
import nd2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import napari
from qtpy.QtCore import Qt
from superqt import QRangeSlider
from PyQt5.QtWidgets import QPushButton, QMainWindow, QLineEdit, QLabel
from PyQt5.QtGui import QFont
import matplotlib.gridspec as gridspec
from ...setup import Notebook
from skimage.filters import sobel
from coppafish.register.preprocessing import n_matches_to_frac_matches, yxz_to_zyx_affine, yxz_to_zyx
from coppafish.register.base import huber_regression, brightness_scale, ols_regression
from coppafish.utils import tiles_io
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
        self.transform = nb.register.initial_transform
        self.z_scale = nbp_basic.pixel_size_z / nbp_basic.pixel_size_xy
        self.r_ref, self.c_ref = nbp_basic.anchor_round, nb.basic_info.anchor_channel
        if nb.get_config()['register']['round_registration_channel'] is None:
            self.round_registration_channel = nbp_basic.anchor_channel
        else:
            self.round_registration_channel = nb.get_config()['register']['round_registration_channel']
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
        # Set default lower limit to 0 and upper limit to 255
        self.im_contrast_limits_slider.setValue((0, 255))
        self.anchor_contrast_limits_slider.setValue((0, 255))

        # Now we run a method that sets these contrast limits using napari
        # Create sliders! We want these sliders to be placed at the top left of the napari viewer
        # Add these sliders as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.im_contrast_limits_slider, area="left", name='Imaging Contrast')
        self.viewer.window.add_dock_widget(self.anchor_contrast_limits_slider, area="left", name='Anchor Contrast')
        # Now create events that will recognise when someone has changed slider values
        self.anchor_contrast_limits_slider.valueChanged.connect(lambda x:
                                                                self.change_anchor_layer_contrast(x[0], x[1]))
        self.im_contrast_limits_slider.valueChanged.connect(lambda x: self.change_imaging_layer_contrast(x[0], x[1]))

        # Add a single button to turn off the base images and a single button to turn off the target images
        self.switch_buttons = ButtonOnOffWindow()
        # This allows us to link clickng to slot functions
        self.switch_buttons.button_base.clicked.connect(self.button_base_images_clicked)
        self.switch_buttons.button_target.clicked.connect(self.button_target_images_clicked)
        # Add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.switch_buttons, area="left", name='Switch', add_vertical_stretch=False)

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
            self.svr_buttons.__getattribute__('R' + str(rnd)).clicked.connect(self.create_round_slot(rnd))
        # Add buttons for correlation coefficients for both hist or cmap
        self.svr_buttons.pearson_hist.clicked.connect(self.button_pearson_hist_clicked)
        self.svr_buttons.pearson_cmap.clicked.connect(self.button_pearson_cmap_clicked)
        # add buttons for spatial correlation coefficients for rounds or channels
        self.svr_buttons.pearson_spatial_round.clicked.connect(self.button_pearson_spatial_round_clicked)
        # Finally, add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.svr_buttons, area="left", name='SVR Diagnostics',
                                           add_vertical_stretch=False)

        # Create a single widget containing buttons for cross tile outlier removal
        # self.outlier_buttons = ButtonOutlierWindow()
        # Now connect buttons to functions
        # self.outlier_buttons.button_vec_field_r.clicked.connect(self.button_vec_field_r_clicked)
        # self.outlier_buttons.button_vec_field_c.clicked.connect(self.button_vec_field_c_clicked)
        # self.outlier_buttons.button_shift_cmap_r.clicked.connect(self.button_shift_cmap_r_clicked)
        # self.outlier_buttons.button_shift_cmap_c.clicked.connect(self.button_shift_cmap_c_clicked)
        # self.outlier_buttons.button_scale_r.clicked.connect(self.button_scale_r_clicked)
        # self.outlier_buttons.button_scale_c.clicked.connect(self.button_scale_c_clicked)
        #
        # Finally, add these buttons as widgets in napari viewer
        # self.viewer.window.add_dock_widget(self.outlier_buttons, area="left", name='Cross Tile Outlier Removal',
        #                                    add_vertical_stretch=False)

        # Create a single widget containing buttons for ICP diagnostics
        self.icp_buttons = ButtonICPWindow()
        # Now connect buttons to functions
        self.icp_buttons.button_mse.clicked.connect(self.button_mse_clicked)
        self.icp_buttons.button_matches.clicked.connect(self.button_matches_clicked)
        self.icp_buttons.button_deviations.clicked.connect(self.button_deviations_clicked)

        # Finally, add these buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.icp_buttons, area="left", name='ICP Diagnostics',
                                           add_vertical_stretch=False)

        # Create a single widget containing buttons for Overlay diagnostics
        self.overlay_buttons = ButtonOverlayWindow()
        # Now connect button to function
        self.overlay_buttons.button_overlay.clicked.connect(self.view_button_clicked)
        # Add buttons as widgets in napari viewer
        self.viewer.window.add_dock_widget(self.overlay_buttons, area="left", name='Overlay Diagnostics',
                                           add_vertical_stretch=False)

        # Create a single widget containing buttons for BG Subtraction diagnostics if bg subtraction has been run
        if self.nb.basic_info.use_preseq:
            self.bg_sub_buttons = ButtonBGWindow()
            # Now connect buttons to function
            self.bg_sub_buttons.button_overlay.clicked.connect(self.button_bg_sub_overlay_clicked)
            self.bg_sub_buttons.button_brightness_scale.clicked.connect(self.button_brightness_scale_clicked)
            # Add buttons as widgets in napari viewer
            self.viewer.window.add_dock_widget(self.bg_sub_buttons, area="left", name='BG Subtraction Diagnostics',
                                               add_vertical_stretch=False)

        # Create a widget containing buttons for fluorescent bead diagnostics if fluorescent beads have been used
        if self.nb.file_names.fluorescent_bead_path is not None:
            self.bead_buttons = ButtonBeadWindow()
            # Now connect buttons to function
            self.bead_buttons.button_fluorescent_beads.clicked.connect(self.button_fluorescent_beads_clicked)
            # Add buttons as widgets in napari viewer
            self.viewer.window.add_dock_widget(self.bead_buttons, area="left", name='Fluorescent Bead Diagnostics',
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

    def button_base_images_clicked(self):
        n_images = len(self.viewer.layers)
        for i in range(0, n_images, 2):
            self.viewer.layers[i].visible = self.switch_buttons.button_base.isChecked()

    def button_target_images_clicked(self):
        n_images = len(self.viewer.layers)
        for i in range(1, n_images, 2):
            self.viewer.layers[i].visible = self.switch_buttons.button_target.isChecked()

    # Contrast
    def change_anchor_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(0, len(self.viewer.layers), 2):
            self.viewer.layers[i].contrast_limits = [low, high]

    # contrast
    def change_imaging_layer_contrast(self, low, high):
        # Change contrast of anchor image (displayed in red), these are even index layers
        for i in range(1, len(self.viewer.layers), 2):
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
            self.transform = self.nb.register.initial_transform
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
        view_pearson_colourmap_spatial(nb=self.nb, t=self.tile)

    # SVR
    def create_round_slot(self, r):

        def round_button_clicked():
            use_rounds = self.nb.basic_info.use_rounds
            for rnd in use_rounds:
                self.svr_buttons.__getattribute__('R' + str(rnd)).setChecked(rnd == r)
            # We don't need to update the plot, we just need to call the viewing function
            view_round_regression_scatter(nb=self.nb, t=self.tile, r=r)

        return round_button_clicked

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

    #  overlay
    def view_button_clicked(self):
        # This function is called when the view button is clicked
        # Need to get the tile, round, channel and filter from the GUI. Then run view_entire_overlay
        # Get the tile, round, channel and filter from the GUI
        t_view = int(self.overlay_buttons.textbox_tile.text())
        r_view = int(self.overlay_buttons.textbox_round.text())
        c_view = int(self.overlay_buttons.textbox_channel.text())
        filter = self.overlay_buttons.button_filter.isChecked()
        # Run view_entire_overlay.
        try:
            view_entire_overlay(nb=self.nb, t=t_view, r=r_view, c=c_view, filter=filter)
        except:
            print('Error: could not view overlay')
        # Reset view button to unchecked and filter button to unchecked
        self.overlay_buttons.button_overlay.setChecked(False)
        self.overlay_buttons.button_filter.setChecked(False)

    # bg subtraction
    def button_bg_sub_overlay_clicked(self):
        t_view = int(self.bg_sub_buttons.textbox_tile.text())
        r_view = int(self.bg_sub_buttons.textbox_round.text())
        c_view = int(self.bg_sub_buttons.textbox_channel.text())
        # Run view_background_overlay.
        try:
            view_background_overlay(nb=self.nb, t=t_view, r=r_view, c=c_view)
        except:
            print('Error: could not view overlay')
        # Reset view button to unchecked and filter button to unchecked
        self.bg_sub_buttons.button_overlay.setChecked(False)

    # bg subtraction
    def button_brightness_scale_clicked(self):
        t_view = int(self.bg_sub_buttons.textbox_tile.text())
        r_view = int(self.bg_sub_buttons.textbox_round.text())
        c_view = int(self.bg_sub_buttons.textbox_channel.text())
        # Run view_background_overlay.
        try:
            view_background_brightness_correction(nb=self.nb, t=t_view, r=r_view, c=c_view)
        except:
            print('Error: could not view brightness scale')
        # Reset view button to unchecked and filter button to unchecked
        self.bg_sub_buttons.button_brightness_scale.setChecked(False)

    # fluorescent beads
    def button_fluorescent_beads_clicked(self):
        try:
            view_camera_correction(nb=self.nb)
        except:
            print('Error: could not view fluorescent beads')
        # Reset view button to unchecked and filter button to unchecked
        self.bead_buttons.button_fluorescent_beads.setChecked(False)

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
            # when we delete layer 0, layer 1 becomes l0 and so on
            del self.viewer.layers[0]

    def get_images(self):
        # reset initial target image lists to empty lists
        use_rounds, use_channels = self.nb.basic_info.use_rounds, self.nb.basic_info.use_channels
        self.target_round_image, self.target_channel_image = [], []
        t = self.tile
        # populate target arrays
        for r in use_rounds:
            file = 't' + str(t) + 'r' + str(r) + 'c' + str(self.round_registration_channel) + '.npy'
            affine = yxz_to_zyx_affine(A=self.transform[t, r, self.c_ref],
                                       new_origin=self.new_origin)
            # Reset the spline interpolation order to 1 to speed things up
            self.target_round_image.append(affine_transform(np.load(os.path.join(self.output_dir, file)),
                                                            affine, order=1))

        for c in use_channels:
            file = 't' + str(t) + 'r' + str(self.r_mid) + 'c' + str(c) + '.npy'
            affine = yxz_to_zyx_affine(A=self.transform[t, self.r_mid, c], new_origin=self.new_origin)
            self.target_channel_image.append(affine_transform(np.load(os.path.join(self.output_dir, file)),
                                                              affine, order=1))
        # populate anchor image
        base_file = 't' + str(t) + 'r' + str(self.r_ref) + 'c' + str(self.round_registration_channel) + '.npy'
        self.base_image_dapi = np.load(os.path.join(self.output_dir, base_file))
        base_file_anchor = 't' + str(t) + 'r' + str(self.r_ref) + 'c' + str(self.c_ref) + '.npy'
        self.base_image = np.load(os.path.join(self.output_dir, base_file_anchor))

    def plot_images(self):
        use_rounds, use_channels = self.nb.basic_info.use_rounds, self.nb.basic_info.use_channels

        # We will add a point on top of each image and add features to it
        features = {'r': np.repeat(np.append(use_rounds, np.ones(len(use_channels)) * self.r_mid), 10).astype(int),
                    'c': np.repeat(np.append(np.ones(len(use_rounds)) * self.round_registration_channel,
                                             use_channels), 10).astype(int)}
        features_anchor = {'r': np.repeat(np.ones(len(use_rounds) + len(use_channels)) * self.r_ref, 10).astype(int),
                           'c': np.repeat(np.ones(len(use_rounds) +
                                                  len(use_channels)) * self.round_registration_channel, 10).astype(int)}

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
            self.viewer.add_image(self.base_image_dapi, blending='additive', colormap='red',
                                  translate=[0, 0, 1_000 * r],
                                  name='Anchor')
            self.viewer.add_image(self.target_round_image[r], blending='additive', colormap='green',
                                  translate=[0, 0, 1_000 * r],
                                  name='Round ' + str(r) + ', Channel ' + str(self.round_registration_channel))
            # Add this to all z planes so still shows up when scrolling
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


class ButtonOnOffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.button_base = QPushButton('Base', self)
        self.button_base.setCheckable(True)
        self.button_base.setGeometry(75, 2, 50, 28)

        self.button_target = QPushButton('Target', self)
        self.button_target.setCheckable(True)
        self.button_target.setGeometry(140, 2, 50, 28)


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

        # Create 2 correlation buttons:
        # 1 to view pearson correlation coefficient as histogram
        # 2 to view pearson correlation coefficient as colormap
        y = y_r + 1
        button = QPushButton('Shift Score \n Hist', self)
        button.setCheckable(True)
        button.setGeometry(0, 40 + 60 * y, 120, 56)
        self.pearson_hist = button
        button = QPushButton('Shift Score \n c-map', self)
        button.setCheckable(True)
        button.setGeometry(140, 40 + 60 * y, 120, 56)
        self.pearson_cmap = button

        # Create 2 spatial correlation buttons:
        # 1 to view pearson correlation coefficient spatially for rounds
        # 2 to view pearson correlation coefficient spatially for channels
        y += 1
        button = QPushButton('Round Score \n Spatial c-map', self)
        button.setCheckable(True)
        button.setGeometry(0, 68 + 60 * y, 120, 56)
        self.pearson_spatial_round = button


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


class ButtonOverlayWindow(QMainWindow):
    # This class creates a window with buttons for viewing overlays
    # We want text boxes for entering the tile, round and channel. We then want a simple button for filtering and
    # for viewing the overlay
    def __init__(self):
        super().__init__()

        self.button_overlay = QPushButton('View', self)
        self.button_overlay.setCheckable(True)
        self.button_overlay.setGeometry(20, 20, 220, 28)

        # Add title to each textbox
        label_tile = QLabel(self)
        label_tile.setText('Tile')
        label_tile.setGeometry(20, 70, 100, 28)
        self.textbox_tile = QLineEdit(self)
        self.textbox_tile.setFont(QFont('Arial', 8))
        self.textbox_tile.setText('0')
        self.textbox_tile.setGeometry(20, 100, 100, 28)

        label_round = QLabel(self)
        label_round.setText('Round')
        label_round.setGeometry(140, 70, 100, 28)
        self.textbox_round = QLineEdit(self)
        self.textbox_round.setFont(QFont('Arial', 8))
        self.textbox_round.setText('0')
        self.textbox_round.setGeometry(140, 100, 100, 28)

        label_channel = QLabel(self)
        label_channel.setText('Channel')
        label_channel.setGeometry(20, 130, 100, 28)
        self.textbox_channel = QLineEdit(self)
        self.textbox_channel.setFont(QFont('Arial', 8))
        self.textbox_channel.setText('18')
        self.textbox_channel.setGeometry(20, 160, 100, 28)

        self.button_filter = QPushButton('Filter', self)
        self.button_filter.setCheckable(True)
        self.button_filter.setGeometry(140, 160, 100, 28)


class ButtonBGWindow(QMainWindow):
    """
    This class creates a window with buttons for viewing background images overlayed with foreground images
    """

    def __init__(self):
        super().__init__()
        self.button_overlay = QPushButton('View Overlay', self)
        self.button_overlay.setCheckable(True)
        self.button_overlay.setGeometry(20, 20, 220, 28)

        self.button_brightness_scale = QPushButton('View BG Scale', self)
        self.button_brightness_scale.setCheckable(True)
        self.button_brightness_scale.setGeometry(20, 70, 220, 28)

        # Add title to each textbox
        label_tile = QLabel(self)
        label_tile.setText('Tile')
        label_tile.setGeometry(20, 130, 100, 28)
        self.textbox_tile = QLineEdit(self)
        self.textbox_tile.setFont(QFont('Arial', 8))
        self.textbox_tile.setText('0')
        self.textbox_tile.setGeometry(20, 160, 100, 28)

        label_round = QLabel(self)
        label_round.setText('Round')
        label_round.setGeometry(140, 130, 100, 28)
        self.textbox_round = QLineEdit(self)
        self.textbox_round.setFont(QFont('Arial', 8))
        self.textbox_round.setText('0')
        self.textbox_round.setGeometry(140, 160, 100, 28)

        label_channel = QLabel(self)
        label_channel.setText('Channel')
        label_channel.setGeometry(20, 190, 100, 28)
        self.textbox_channel = QLineEdit(self)
        self.textbox_channel.setFont(QFont('Arial', 8))
        self.textbox_channel.setText('18')
        self.textbox_channel.setGeometry(20, 220, 100, 28)


class ButtonBeadWindow(QMainWindow):
    """
    This class creates a window with buttons for viewing fluorescent bead images
    """

    def __init__(self):
        super().__init__()
        self.button_fluorescent_beads = QPushButton('View Fluorescent Beads', self)
        self.button_fluorescent_beads.setCheckable(True)
        self.button_fluorescent_beads.setGeometry(20, 20, 220, 28)


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
def view_round_regression_scatter(nb: Notebook, t: int, r: int):
    """
    view 9 scatter plots for each data set shift vs positions
    Args:
        nb: Notebook
        t: tile
        r: round
    """
    # Transpose shift and position variables so coord is dimension 0, makes plotting easier
    shift = nb.register_debug.round_shift[t, r]
    corr = nb.register_debug.round_shift_corr[t, r]
    position = nb.register_debug.position[t, r]
    initial_transform = nb.register_debug.round_transform_raw[t, r]
    icp_transform = yxz_to_zyx_affine(A=nb.register.transform[t, r, nb.basic_info.anchor_channel])

    r_thresh = nb.get_config()['register']['pearson_r_thresh']
    shift = shift[corr > r_thresh].T
    position = position[corr > r_thresh].T

    # Make ranges, wil be useful for plotting lines
    z_range = np.arange(np.min(position[0]), np.max(position[0]))
    yx_range = np.arange(np.min(position[1]), np.max(position[1]))
    coord_range = [z_range, yx_range, yx_range]
    # Need to add a central offset to all lines plotted
    tile_centre_zyx = np.roll(nb.basic_info.tile_centre, 1)

    # We want to plot the shift of each coord against the position of each coord. The gradient when the dependent var
    # is coord i and the independent var is coord j should be the transform[i,j] - int(i==j)
    gradient_svr = initial_transform[:3, :3] - np.eye(3)
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
            central_offset_svr[i, j] = gradient_svr[i, k1] * tile_centre_zyx[k1] + gradient_svr[i, k2] * \
                                       tile_centre_zyx[k2]
            central_offset_icp[i, j] = gradient_icp[i, k1] * tile_centre_zyx[k1] + gradient_icp[i, k2] * \
                                       tile_centre_zyx[k2]
            # Now compute the intercepts
            intercpet_svr[i, j] = initial_transform[i, 3] + central_offset_svr[i, j]
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
    round_registration_channel = nb.get_config()['register']['round_registration_channel']
    if round_registration_channel is None:
        round_registration_channel = nb.basic_info.anchor_channel
    plt.suptitle('Round regression for Tile ' + str(t) + ', Round ' + str(r) + 'Channel '
                 + str(round_registration_channel))
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
    round_corr = nbp_register_debug.round_shift_corr[t]
    n_rounds = nbp_basic.n_rounds
    cols = n_rounds

    for r in range(n_rounds):
        plt.subplot(1, cols, r + 1)
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
    round_corr = nbp_register_debug.round_shift_corr[t]

    # Replace 0 with nans so they get plotted as black
    round_corr[round_corr == 0] = np.nan
    # plot round correlation
    fig, ax = plt.subplots(1, 1)
    # ax1 refers to round shifts
    im = ax.imshow(round_corr, vmin=0, vmax=1, aspect='auto', interpolation='none')
    ax.set_xlabel('Sub-volume index')
    ax.set_ylabel('Round')
    ax.set_title('Round sub-volume shift scores')

    # Add common colour bar. Also give it the label 'Pearson correlation coefficient'
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Correlation coefficient')

    plt.suptitle('Similarity score distributions for all sub-volume shifts')


# 1
def view_pearson_colourmap_spatial(nb: Notebook, t: int):
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
    use = nb.basic_info.use_rounds
    corr = nb.register_debug.round_shift_corr[t, use]
    mode = 'Round'

    # Set 0 correlations to nan, so they are plotted as black
    corr[corr == 0] = np.nan
    z_subvols, y_subvols, x_subvols = config['subvols']
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
def shift_vector_field(nb: Notebook):
    """
    Function to plot vector fields of predicted shifts vs shifts to see if we classify a shift as an outlier.
    Args:
        nb: Notebook
        round: True if round, False if Channel
    """
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    residual_thresh = nb.get_config()['register']['residual_thresh']
    use_tiles = nbp_basic.use_tiles
    shift = nbp_register_debug.round_transform_raw[use_tiles, :, :, 3]
    # record number of rounds, tiles and initialise predicted shift
    n_tiles, n_rounds = shift.shape[0], nbp_basic.n_rounds
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]
    tilepos_yx_padded = np.vstack((tilepos_yx.T, np.ones(n_tiles))).T
    predicted_shift = np.zeros_like(shift)
    # When we are scaling the vector field, it will be useful to store the following
    n_vectors_x = tilepos_yx[:, 1].max() - tilepos_yx[:, 1].min() + 1
    shift_norm = np.linalg.norm(shift, axis=2)

    fig, axes = plt.subplots(nrows=3, ncols=n_rounds)
    for r in range(n_rounds):
        # generate predicted shift for this round
        lb, ub = np.percentile(shift_norm[:, r], [10, 90])
        valid = (shift_norm[:, r] > lb) * (shift_norm[:, r] < ub)
        # Carry out regression, first predicitng yx shift, then z shift
        transform_yx = np.linalg.lstsq(tilepos_yx_padded[valid], shift[valid, r, 1:], rcond=None)[0]
        predicted_shift[:, r, 1:] = tilepos_yx_padded @ transform_yx
        transform_z = np.linalg.lstsq(tilepos_yx_padded[valid], shift[valid, r, 0][:, None], rcond=None)[0]
        predicted_shift[:, r, 0] = (tilepos_yx_padded @ transform_z)[:, 0]
        # Defining this scale will mean that the length of the largest vector will be equal to 1/n_vectors_x of the
        # width of the plot
        scale = n_vectors_x * np.sqrt(np.sum(predicted_shift[:, r, 1:] ** 2, axis=1))

        # plot the predicted yx shift vs actual yx shift in row 0
        ax = axes[0, r]
        # Make sure the vector field is properly scaled
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], predicted_shift[:, r, 2], predicted_shift[:, r, 1],
                  color='b', scale=scale, scale_units='width', width=.05, alpha=0.5)
        ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], shift[:, r, 2], shift[:, r, 1], color='r', scale=scale,
                  scale_units='width', width=.05, alpha=0.5)
        # We want to set the xlims and ylims to include a bit of padding so we can see the vectors
        ax.set_xlim(tilepos_yx[:, 1].min() - 1, tilepos_yx[:, 1].max() + 1)
        ax.set_ylim(tilepos_yx[:, 0].min() - 1, tilepos_yx[:, 0].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # plot the predicted z shift vs actual z shift in row 1
        ax = axes[1, r]
        # we only want 1 label so make this for r = 0
        if r == 0:
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, predicted_shift[:, r, 0], color='b', scale=scale,
                      scale_units='width', width=.05, alpha=0.5, label='Predicted')
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, shift[:, r, 2], color='r', scale=scale,
                      scale_units='width', width=.05, alpha=0.5, label='Actual')
        else:
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, predicted_shift[:, r, 0], color='b', scale=scale,
                      scale_units='width', width=.05, alpha=0.5)
            ax.quiver(tilepos_yx[:, 1], tilepos_yx[:, 0], 0, shift[:, r, 2], color='r', scale=scale,
                      scale_units='width', width=.05, alpha=0.5)
        # We want to set the xlims and ylims to include a bit of padding so we can see the vectors
        ax.set_xlim(tilepos_yx[:, 1].min() - 1, tilepos_yx[:, 1].max() + 1)
        ax.set_ylim(tilepos_yx[:, 0].min() - 1, tilepos_yx[:, 0].max() + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot image of norms of residuals at each tile in row 3
        ax = axes[2, r]
        diff = create_tiled_image(data=np.linalg.norm(predicted_shift[:, r] - shift[:, r], axis=1),
                                  nbp_basic=nbp_basic)
        outlier = np.argwhere(diff > residual_thresh)
        n_outliers = outlier.shape[0]
        im = ax.imshow(diff, vmin=0, vmax=10)
        # Now we want to outline the outlier pixels with a dotted red rectangle
        for i in range(n_outliers):
            rect = patches.Rectangle((outlier[i, 1] - 0.5, outlier[i, 0] - 0.5), 1, 1, linewidth=1,
                                     edgecolor='r', facecolor='none', linestyle='--')
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_yticks([])

    # Set row and column labels
    for ax, col in zip(axes[0], nbp_basic.use_rounds):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], ['XY-shifts', 'Z-shifts', 'Residuals']):
        ax.set_ylabel(row, rotation=90, size='large')
    # Add row and column labels
    fig.supxlabel('Round')
    fig.supylabel('Diagnostic')
    # Add global colour bar and legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Residual Norm')
    # Add title
    fig.suptitle('Diagnostic plots for round shift outlier removal', size='x-large')


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
        shift_raw = nbp_register_debug.round_transform_raw[use_tiles, :, :, 3]
        shift = nbp_register.round_transform[use_tiles, :, :, 3]
    else:
        mode = 'Channel'
        use = nbp_basic.use_channels
        shift_raw = nbp_register_debug.channel_transform_raw[use_tiles, :, :, 3]
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
        diff[tilepos_yx[t, 0], tilepos_yx[t, 1]] = data[t]

    diff[diff == 0] = np.nan

    return diff


# 2
def view_round_scales(nb: Notebook):
    """
    view scale parameters for the round outlier removals
    Args:
        nb: Notebook
    """
    # TODO: This is showing a bug in the code that all unregularised transforms have same z scale
    nbp_basic, nbp_register_debug = nb.basic_info, nb.register_debug
    anchor_round, anchor_channel = nbp_basic.anchor_round, nbp_basic.anchor_channel
    use_tiles = nbp_basic.use_tiles
    # Extract raw scales
    z_scale = nbp_register_debug.round_transform_raw[use_tiles, :, 0, 0]
    y_scale = nbp_register_debug.round_transform_raw[use_tiles, :, 1, 1]
    x_scale = nbp_register_debug.round_transform_raw[use_tiles, :, 2, 2]
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

    plt.suptitle('Distribution of scales across tiles for round registration.')
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
    z_scale = nbp_register_debug.channel_transform_raw[use_tiles][:, use_channels, 0, 0]
    y_scale = nbp_register_debug.channel_transform_raw[use_tiles][:, use_channels, 1, 1]
    x_scale = nbp_register_debug.channel_transform_raw[use_tiles][:, use_channels, 2, 2]
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

    plt.suptitle('Distribution of scales across tiles for channel registration.')
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
    # delete column 1 from n_matches as it is incorrect
    n_matches = np.delete(n_matches, 1, axis=2)
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
            ax.set_yticks([])
            ax.set_ylim([0, 1])
            ax.set_xlim([0, n_iters // 2])

    plt.suptitle('Fraction of matches against iterations for tile ' + str(t) + '. \n '
                                                                               'Note that y-axis is [0,1]')
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
    # delete column 1 from mse as it is incorrect
    mse = np.delete(mse, 1, axis=2)
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
            ax.set_yticks([])
            ax.set_xlim([0, n_iters // 2])
            ax.set_ylim([0, np.max(mse)])

    plt.suptitle('MSE against iteration for tile ' + str(t) + ' for all rounds and channels. \n'
                                                              'Note that the y-axis is the same for all plots.')
    plt.show()


# 3
def view_icp_deviations(nb: Notebook, t: int):
    """
    Plots deviations of ICP transform for a given tile t (n_rounds x n_channel x 3 x 4) affine transform against initial
    guess (initial_transform) which has the same shape. These trasnforms are in zyx x zyx format, with the final col
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
    initial_transform = np.zeros((len(use_rounds), len(use_channels), 3, 4))
    transform = np.zeros((len(use_rounds), len(use_channels), 3, 4))
    for r in range(len(use_rounds)):
        for c in range(len(use_channels)):
            initial_transform[r, c] = yxz_to_zyx_affine(
                A=nbp_register.initial_transform[t, use_rounds[r], use_channels[c]])
            transform[r, c] = yxz_to_zyx_affine(A=nbp_register.transform[t, use_rounds[r], use_channels[c]])

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
            scale_diff[r, c] = np.diag(transform[r, c, :3, :3]) - np.diag(initial_transform[r, c, :3, :3])
            shift_diff[r, c] = transform[r, c, :3, 3] - initial_transform[r, c, :3, 3]

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
    fig.subplots_adjust(right=0.7)
    cbar_scale_ax = fig.add_axes([0.75, 0.15, 0.025, 0.7])
    # Next we want to make sure the scale cbar has ticks on the left.
    cbar_scale_ax.yaxis.tick_left()
    fig.colorbar(im1, cax=cbar_scale_ax, ticks=[-.1, 0, .1], label='Scale difference')
    cbar_shift_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
    fig.colorbar(im2, cax=cbar_shift_ax, ticks=[-5, 0, 5], label='Shift difference')

    plt.suptitle('Deviations of ICP from SVR for tile ' + str(t) + '. \n'
                                                                   'Left column is zyx scale difference, '
                                                                   'right column is zyx shift difference.')


def view_entire_overlay(nb: Notebook, t: int, r: int, c: int, filter=False):
    """
    Plots the entire image for a given tile t, round r, channel c, and overlays the SVR transformed image on top of the
    ICP transformed image. The SVR transformed image is in red, and the ICP transformed image is in green.

    Args:
        nb: Notebook
        t: tile
        r: round
        c: channel
        filter: whether to apply sobel filter to images
    """
    # Initialise frequent variables
    anchor = yxz_to_zyx(
        tiles_io.load_tile(nb.file_names, nb.basic_info, nb.extract.file_type, t, nb.basic_info.anchor_round,
                           nb.basic_info.anchor_channel))
    target = yxz_to_zyx(tiles_io.load_tile(nb.file_names, nb.basic_info, nb.extract.file_type, t, r, c))
    transform = yxz_to_zyx_affine(nb.register.transform[t, r, c])
    target_transfromed = affine_transform(target, transform, order=1)
    # plot in napari
    if filter:
        anchor = sobel(anchor)
        target = sobel(target)
        target_transfromed = sobel(target_transfromed)

    viewer = napari.Viewer()
    viewer.add_image(anchor, name='Tile ' + str(t) + ', round ' + str(nb.basic_info.anchor_round) + ', channel ' +
                                  str(nb.basic_info.anchor_channel), colormap='red', blending='additive')
    viewer.add_image(target_transfromed, name='Tile ' + str(t) + ', round ' + str(r) + ', channel ' + str(c) +
                                              ' transformed', colormap='green', blending='additive')
    viewer.add_image(target, name='Tile ' + str(t) + ', round ' + str(r) + ', channel ' + str(c), colormap='blue',
                     blending='additive', opacity=0)
    napari.run()


def view_background_overlay(nb: Notebook, t: int, r: int, c: int):
    """
    Overlays tile t, round r, channel c with the preseq image for the same tile, and the same channel. The preseq image
    is in red, and the seq image is in green. Both are registered.
    Args:
        nb: Notebook
        t: tile
        r: round
        c: channel
    """

    if c in nb.basic_info.use_channels:
        transform_preseq = yxz_to_zyx_affine(nb.register.transform[t, nb.basic_info.pre_seq_round, c])
        transform_seq = yxz_to_zyx_affine(nb.register.transform[t, r, c])
    elif c == 0:
        transform_preseq = yxz_to_zyx_affine(nb.register.round_transform[t, nb.basic_info.pre_seq_round])
        if r == nb.basic_info.anchor_round:
            transform_seq = np.eye(4)
        else:
            transform_seq = yxz_to_zyx_affine(nb.register.round_transform[t, r])
    seq = yxz_to_zyx(tiles_io.load_tile(nb.file_names, nb.basic_info, nb.extract.file_type, t, r, c))
    preseq = yxz_to_zyx(
        tiles_io.load_tile(nb.file_names, nb.basic_info, nb.extract.file_type, t, nb.basic_info.pre_seq_round, c))

    print('Starting Application of Seq Transform')
    seq = affine_transform(seq, transform_seq, order=1)
    print('Starting Application of Preseq Transform')
    preseq = affine_transform(preseq, transform_preseq, order=1)
    print('Finished Transformations')

    viewer = napari.Viewer()
    viewer.add_image(seq, name='seq', colormap='green', blending='additive')
    viewer.add_image(preseq, name='preseq', colormap='red', blending='additive')
    napari.run()


def view_background_brightness_correction(nb: Notebook, t: int, r: int, c: int, percentile: int = 99,
                                          sub_image_size: int = 500, bg_blur: bool = True):
    print(f"Computing background scale for tile {t}, round {r}, channel {c}")
    num_z = nb.basic_info.tile_centre[2].astype(int)
    transform_pre = yxz_to_zyx_affine(nb.register.transform[t, nb.basic_info.pre_seq_round, c],
                                      new_origin=np.array([num_z - 5, 0, 0]))
    transform_seq = yxz_to_zyx_affine(nb.register.transform[t, r, c], new_origin=np.array([num_z - 5, 0, 0]))
    preseq = yxz_to_zyx(
        tiles_io.load_tile(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=nb.basic_info.pre_seq_round, c=c,
                           yxz=[None, None, np.arange(num_z - 5, num_z + 5)], suffix='_raw' * (1 - bg_blur)))
    seq = yxz_to_zyx(tiles_io.load_tile(nb.file_names, nb.basic_info, nb.extract.file_type, t=t, r=r, c=c,
                                        yxz=[None, None, np.arange(num_z - 5, num_z + 5)]))
    preseq = affine_transform(preseq, transform_pre, order=5)
    seq = affine_transform(seq, transform_seq, order=5)
    # Apply background scale. Don't subtract bg from pixels that are <= 0 in seq as blurring in preseq can cause
    # negative values in seq
    bg_scale, sub_seq, sub_preseq = brightness_scale(preseq, seq, percentile, sub_image_size)
    mask = sub_preseq > np.percentile(sub_preseq, percentile)
    diff = sub_seq - bg_scale * sub_preseq
    ratio = sub_seq[mask] / sub_preseq[mask]
    estimate_scales = np.percentile(ratio, [25, 75])
    diff_low = sub_seq - estimate_scales[0] * sub_preseq
    diff_high = sub_seq - estimate_scales[1] * sub_preseq

    # View overlay and view regression
    viewer = napari.Viewer()
    viewer.add_image(sub_seq, name='seq', colormap='green', blending='additive')
    viewer.add_image(sub_preseq, name='preseq', colormap='red', blending='additive')
    viewer.add_image(diff, name='diff', colormap='gray', blending='translucent', visible=False)
    viewer.add_image(diff_low, name='bg_scale = 25%', colormap='gray', blending='translucent', visible=False)
    viewer.add_image(diff_high, name='bg_scale = 75%', colormap='gray', blending='translucent', visible=False)
    viewer.add_image(mask, name='mask', colormap='blue', blending='additive', visible=False)

    # View regression
    plt.subplot(1, 2, 1)
    bins = 25
    plt.hist2d(sub_preseq[mask], sub_seq[mask], bins=[np.linspace(0, np.percentile(sub_preseq[mask], 90), bins),
                                                      np.linspace(0, np.percentile(sub_seq[mask], 90), bins)])
    x = np.linspace(0, np.percentile(sub_seq[mask], 90), 100)
    y = bg_scale * x
    plt.plot(x, y, 'r')
    plt.plot(x, estimate_scales[0] * x, 'g')
    plt.plot(x, estimate_scales[1] * x, 'g')
    plt.xlabel('Preseq')
    plt.ylabel('Seq')
    plt.title('Regression of preseq vs seq. Scale = ' + str(np.round(bg_scale[0], 3)))

    plt.subplot(1, 2, 2)
    plt.hist(sub_seq[mask] / sub_preseq[mask], bins=100)
    max_bin_val = np.max(np.histogram(sub_seq[mask] / sub_preseq[mask], bins=100)[0])
    plt.vlines(bg_scale, 0, max_bin_val, colors='r')
    plt.vlines(estimate_scales, 0, max_bin_val, colors='g')
    plt.xlabel('Seq / Preseq')
    plt.ylabel('Frequency')
    plt.title('Histogram of seq / preseq. Scale = ' + str(np.round(bg_scale[0], 3)))
    plt.show()

    napari.run()


def view_camera_correction(nb: Notebook):
    """
    Plots the camera correction for each camera against the anchor camera
    Args:
        nb: Notebook (must have register page and a path to fluorescent bead images)
    """
    # One transform for each camera
    viewer = napari.Viewer()
    fluorescent_bead_path = nb.file_names.fluorescent_bead_path
    # open the fluorescent bead images as nd2 files
    with nd2.ND2File(fluorescent_bead_path) as fbim:
        fluorescent_beads = fbim.asarray()

    if len(fluorescent_beads.shape) == 4:
        mid_z = fluorescent_beads.shape[0] // 2
        fluorescent_beads = fluorescent_beads[mid_z, :, :, :]
    # if fluorescent bead images are for all channels, just take one from each camera
    cam_channels = [0, 9, 18, 23]
    if len(fluorescent_beads) == 28:
        fluorescent_beads = fluorescent_beads[cam_channels]
    transform = nb.register.channel_transform[cam_channels][:, 1:, 1:]

    # Apply the transform to the fluorescent bead images
    fluorescent_beads_transformed = np.zeros(fluorescent_beads.shape)
    for c in range(4):
        fluorescent_beads_transformed[c] = affine_transform(fluorescent_beads[c], transform[c], order=3)

    # Add the images to napari
    colours = ['yellow', 'red', 'green', 'blue']
    for c in range(1, 4):
        viewer.add_image(fluorescent_beads[c], name='Camera ' + str(cam_channels[c]), colormap=colours[c],
                         blending='additive', visible=False)
        viewer.add_image(fluorescent_beads_transformed[c], name='Camera ' + str(cam_channels[c]) + ' transformed',
                         colormap=colours[c], blending='additive', visible=True)

    napari.run()