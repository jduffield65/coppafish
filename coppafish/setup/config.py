# Load config files for the Python port of the coppafish pipeline.

# There are three main features of this file:
# 1. Load config from .ini files
# 2. Also load from a "default" .ini file
# 3. Perform validation of the files and the assigned values.

# Config will be available as the result of the function "get_config" in the
# form of a dictionary.  This is, by default, the "Config" variable defined in
# this file.  Since it is a dictionary, access elements of the configuration
# using the subscript operator, i.e. square brackets or item access.  E.g.,
# Config['section']['item'].

# Config files should be considered read-only.

# To add new configuration options, do the following:

# 1. Add it to the "_options" dictionary, defined below.  The name of the
#    configuration option should be the key, and the value should be the
#    "type".  (The types are denied below in the "_option_type_checkers" and
#    "_option_formatters" dictionaries.)
# 2. Add it, and a description of what it does, to "config.default.ini".

import configparser
import copy
import os.path
import os
import re

# List of options and their type.  If you change this, update the
# config.default.ini file too.  Make sure the type is valid.
_options = {
    'basic_info':
        {
            'use_tiles': 'maybe_list_int',
            'use_rounds': 'maybe_list_int',
            'use_channels': 'maybe_list_int',
            'use_z': 'maybe_list_int',
            'use_dyes': 'maybe_list_int',
            'use_anchor': 'bool',
            'anchor_round': 'maybe_int',
            'anchor_channel': 'maybe_int',
            'dapi_channel': 'maybe_int',
            'tile_pixel_value_shift': 'int',
            'dye_names': 'list_str',
            'is_3d': 'bool',
            'ignore_first_z_plane': 'bool',
            # From here onwards these are not compulsory to enter and will be taken from the metadata
            # Only leaving them here to have backwards compatibility as Max thinks the user should influence these
            'channel_camera': 'maybe_list_int',
            'channel_laser': 'maybe_list_int',
            'ref_round': 'maybe_int',
            'ref_channel': 'maybe_int'
        },
    'file_names':
        {
            'notebook_name': 'str',
            'input_dir': 'str',  # all these directories used to be of type 'dir' but you may want to load the notebook
            'output_dir': 'str',  # while not being connected to server where data is
            'tile_dir': 'str',
            'round': 'maybe_list_str',  #
            'anchor': 'maybe_str',
            'raw_extension': 'str',
            'raw_metadata': 'maybe_str',
            'dye_camera_laser': 'maybe_file',
            'code_book': 'str',
            'scale': 'str',
            'spot_details_info': 'str',
            'psf': 'str',
            'omp_spot_shape': 'str',
            'omp_spot_info': 'str',
            'omp_spot_coef': 'str',
            'big_dapi_image': 'maybe_str',
            'big_anchor_image': 'maybe_str',
            'pciseq': 'list_str',
            'fluorescent_bead_path': 'maybe_str',
            'pre_seq': 'maybe_str',
            'initial_bleed_matrix': 'maybe_str'

        },
    'extract':
        {
            'file_type': 'str',
            'wait_time': 'int',
            'r1': 'maybe_int',
            'r2': 'maybe_int',
            'r_dapi': 'maybe_int',
            'r1_auto_microns': 'number',
            'r_dapi_auto_microns': 'maybe_number',
            'scale': 'maybe_number',
            'scale_norm': 'maybe_int',
            'scale_anchor': 'maybe_number',
            'auto_thresh_multiplier': 'number',
            'deconvolve': 'bool',
            'psf_detect_radius_xy': 'int',
            'psf_detect_radius_z': 'int',
            'psf_intensity_thresh': 'maybe_number',
            'psf_isolation_dist': 'number',
            'psf_min_spots': 'int',
            'psf_shape': 'list_int',
            'psf_annulus_width': 'number',
            'wiener_constant': 'number',
            'wiener_pad_shape': 'list_int',
            'r_smooth': 'maybe_list_int',
            'n_clip_warn': 'int',
            'n_clip_error': 'maybe_int',
            'n_clip_error_images_thresh': 'int',
            'num_rotations': 'int',
            'continuous_dapi': 'bool',
            'pre_seq_blur_radius': 'maybe_int'
        },
    'find_spots':
        {
            'radius_xy': 'int',
            'radius_z': 'int',
            'max_spots_2d': 'int',
            'max_spots_3d': 'int',
            'isolation_radius_inner': 'number',
            'isolation_radius_xy': 'number',
            'isolation_radius_z': 'number',
            'isolation_thresh': 'maybe_number',
            'auto_isolation_thresh_multiplier': 'number',
            'n_spots_warn_fraction': 'number',
            'n_spots_error_fraction': 'number'
        },
    'stitch':
        {
            'expected_overlap': 'number',
            'auto_n_shifts': 'list_int',
            'shift_north_min': 'maybe_list_int',
            'shift_north_max': 'maybe_list_int',
            'shift_east_min': 'maybe_list_int',
            'shift_east_max': 'maybe_list_int',
            'shift_step': 'list_int',
            'shift_widen': 'list_int',
            'shift_max_range': 'list_int',
            'neighb_dist_thresh': 'number',
            'shift_score_thresh': 'maybe_number',
            'shift_score_thresh_multiplier': 'number',
            'shift_score_thresh_min_dist': 'number',
            'shift_score_thresh_max_dist': 'number',
            'nz_collapse': 'int',
            'n_shifts_error_fraction': 'number',
            'save_image_zero_thresh': 'int',
            'flip_y': 'bool',
            'flip_x': 'bool'
        },
    'register':
        {
            'neighb_dist_thresh_2d': 'number',
            'neighb_dist_thresh_3d': 'number',
            'subvols': 'list_int',
            'box_size': 'list_int',
            'pearson_r_thresh': 'number',
            'residual_thresh': 'int',
            'bead_radii': 'maybe_list_number',
            'icp_min_spots': 'int',
            'icp_max_iter': 'int',
            'round_registration_channel': 'maybe_int',
            'sobel': 'bool'
        },
    'call_spots':
        {
            'bleed_matrix_method': 'str',
            'bleed_matrix_score_thresh': 'number',
            'bleed_matrix_min_cluster_size': 'int',
            'bleed_matrix_n_iter': 'int',
            'bleed_matrix_anneal': 'bool',
            'background_weight_shift': 'maybe_number',
            'dp_norm_shift': 'maybe_number',
            'norm_shift_min': 'number',
            'norm_shift_max': 'number',
            'norm_shift_precision': 'number',
            'gene_efficiency_min_spots': 'int',
            'gene_efficiency_score_thresh': 'number',
            'gene_efficiency_intensity_thresh': 'number',
            'gene_efficiency_intensity_thresh_percentile': 'number',
            'alpha': 'number',
            'beta': 'number',
        },
    'omp':
        {
            'use_z': 'maybe_list_int',
            'weight_coef_fit': 'bool',
            'initial_intensity_thresh': 'maybe_number',
            'initial_intensity_thresh_percentile': 'int',
            'initial_intensity_thresh_min': 'number',
            'initial_intensity_thresh_max': 'number',
            'initial_intensity_precision': 'number',
            'max_genes': 'int',
            'dp_thresh': 'number',
            'alpha': 'number',
            'beta': 'number',
            'initial_pos_neighbour_thresh': 'maybe_int',
            'initial_pos_neighbour_thresh_param': 'number',
            'initial_pos_neighbour_thresh_min': 'int',
            'initial_pos_neighbour_thresh_max': 'int',
            'radius_xy': 'int',
            'radius_z': 'int',
            'shape_max_size': 'list_int',
            'shape_pos_neighbour_thresh': 'int',
            'shape_isolation_dist': 'number',
            'shape_sign_thresh': 'number'
        },
    'thresholds':
        {
            'intensity': 'maybe_number',
            'score_ref': 'number',
            'score_omp': 'number',
            'score_omp_multiplier': 'number'
        },
    'reg_to_anchor_info':
        {
            'full_anchor_y0': 'maybe_number',
            'full_anchor_x0': 'maybe_number',
            'partial_anchor_y0': 'maybe_number',
            'partial_anchor_x0': 'maybe_number',
            'side_length': 'maybe_number'
        }
}

# If you want to add a new option type, first add a type checker, which will
# only allow valid values to be passed.  Then, add a formatter.  Since the
# config file is strings only, the formatter converts from a string to the
# desired type.  E.g. for the "integer" type, it should be available as an
# integer.
#
# Any new type checkers created should keep in mind that the input is a string,
# and so validation must be done in string form.
#
# "maybe" types come from the Haskell convention whereby it can either hold a
# value or be empty, where empty in this case is defined as an empty string.
# In practice, this means the option is optional.
_option_type_checkers = {
    'int': lambda x: re.match("-?[0-9]+", x) is not None,
    'number': lambda x: re.match("-?[0-9]+(\\.[0-9]+)?$", "-123") is not None,
    'str': lambda x: len(x) > 0,
    'bool': lambda x: re.match("True|true|False|false", x) is not None,
    'file': lambda x: os.path.isfile(x),
    'dir': lambda x: os.path.isdir(x),
    'list': lambda x: True,
    'list_int': lambda x: all([_option_type_checkers['int'](s.strip()) for s in x.split(",")]),
    'list_number': lambda x: all([_option_type_checkers['number'](s.strip()) for s in x.split(",")]),
    'list_str': lambda x: all([_option_type_checkers['str'](s.strip()) for s in x.split(",")]),
    'maybe_int': lambda x: x.strip() == "" or _option_type_checkers['int'](x),
    'maybe_number': lambda x: x.strip() == "" or _option_type_checkers['number'](x),
    'maybe_list_int': lambda x: x.strip() == "" or _option_type_checkers['list_int'](x),
    'maybe_list_number': lambda x: x.strip() == "" or _option_type_checkers['list_number'](x),
    'maybe_str': lambda x: x.strip() == "" or _option_type_checkers['str'](x),
    'maybe_list_str': lambda x: x.strip() == "" or _option_type_checkers['list_str'](x),
    'maybe_file': lambda x: x.strip() == "" or _option_type_checkers['file'](x)
}
_option_formatters = {
    'int': lambda x: int(x),
    'number': lambda x: float(x),
    'str': lambda x: x,
    'bool': lambda x: True if "rue" in x else False,
    'file': lambda x: x,
    'dir': lambda x: x,
    'list': lambda x: [s.strip() for s in x.split(",")],
    'list_int': lambda x: [_option_formatters['int'](s.strip()) for s in x.split(",")],
    'list_number': lambda x: [_option_formatters['number'](s.strip()) for s in x.split(",")],
    'list_str': lambda x: [_option_formatters['str'](s.strip()) for s in x.split(",")],
    'maybe_int': lambda x: None if x == "" else _option_formatters['int'](x),
    'maybe_number': lambda x: None if x == "" else _option_formatters['number'](x),
    'maybe_list_int': lambda x: None if x == "" else _option_formatters['list_int'](x),
    'maybe_list_number': lambda x: None if x == "" else _option_formatters['list_number'](x),
    'maybe_str': lambda x: None if x == "" else _option_formatters['str'](x),
    'maybe_list_str': lambda x: None if x == "" else _option_formatters['list_str'](x),
    'maybe_file': lambda x: None if x == "" else _option_formatters['file'](x)
}


# Standard formatting for errors in the config file
class InvalidConfigError(Exception):
    """Exception for an invalid configuration item"""

    def __init__(self, section, name, val):
        if val is None: val = ""
        if name is None:
            if section in _options.keys():
                error = f"Error in config file: Section {section} must be included in config file"
            else:
                error = f"Error in config file: {section} is not a valid section"
        else:
            if name in _options[section].keys():
                error = f"Error in config file: {name} in section {section} must be a {_options[section][name]}," \
                        f" but the current value {val!r} is not."
            else:
                error = f"Error in config file: {name} in section {section} is not a valid configuration key," \
                        f" and should not exist in the config file. (It is currently set to value {val!r}.)"
        super().__init__(error)


def get_config(ini_file):
    """Return the configuration as a dictionary"""
    # Read the settings files, overwriting the default settings with any settings
    # in the user-editable settings file.  We use .ini files without sections, and
    # add the section (named "config") manually.
    _parser = configparser.ConfigParser()
    _parser.optionxform = str  # Make names case-sensitive
    ini_file_default = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'settings.default.ini')
    with open(ini_file_default, 'r') as f:
        _parser.read_string(f.read())
    # Try to autodetect whether the user has passed a config file or the full
    # text of the config file.  The easy way would be to use os.path.isfile to
    # check if it is a file, and if not, assume it is text.  However, this
    # could lead to confusing error messages.  Instead, we will use the
    # following procedure.  If the string contains only whitespace, assume it
    # is full text.  If it doesn't have a newline or an equal sign, assume it
    # is a path.
    if ini_file.strip() != "" and "=" not in ini_file and "\n" not in ini_file:
        with open(ini_file, 'r') as f:
            _parser.read_string(f.read())
    else:
        _parser.read_string(ini_file)

    # Validate configuration.
    # First step: ensure two things...
    # 1. ensure all of the sections (defined in _options) included
    for section in _options.keys():
        if section not in _parser.keys():
            raise InvalidConfigError(section, None, None)
    # 2. ensure all of the options in each section (defined in
    # _options) have some value.
    for section in _options.keys():
        for name in _options[section].keys():
            if name not in _parser[section].keys():
                raise InvalidConfigError(section, name, None)
    # Second step of validation: ensure three things...
    ini_file_sections = list(_parser.keys())
    ini_file_sections.remove('DEFAULT')  # parser always contains this key.
    # 1. Ensure there are no extra sections in config file
    for section in ini_file_sections:
        if section not in _options.keys():
            raise InvalidConfigError(section, None, None)
    for section in _options.keys():
        for name, val in _parser[section].items():
            # 2. Ensure there are no extra options in config file, else remove them
            if name not in _options[section].keys():
                _parser[section].__delitem__(name)
                continue
            # 3. Ensure that all the option values pass type checking.
            if not _option_type_checkers[_options[section][name]](val):
                raise InvalidConfigError(section, name, val)

    # Now that we have validated, build the configuration dictionary
    out_dict = {section: {} for section in _options.keys()}
    for section in _options.keys():
        for name, val in _parser[section].items():
            out_dict[section][name] = _option_formatters[_options[section][name]](_parser[section][name])
    return out_dict


def split_config(config_file):
    """
    This function will be used to split the config file into one config file for each tile when in parallel mode.

    Args:
        config_file: Path to global config file given.
    Returns:
        config_file_path: (n_tiles) path to each config file.
    """
    config_file_path = []
    # We will create the various config files for each tile. It is easier to read info from the config dict than the
    # ini file so use this where necessary
    config_dict = get_config(config_file)
    use_tiles = config_dict['basic_info']['use_tiles']

    # Create the output directory for the parallel run
    par_dir = os.path.join(config_dict['file_names']['output_dir'], 'par')
    if not os.path.exists(par_dir):
        os.mkdir(par_dir)

    for t in use_tiles:
        # Need to load in the config file
        cfg = configparser.ConfigParser()
        cfg.read(config_file)

        # Need to change the file_names section.
        # First create new folder for each tile notebook
        tile_dir = os.path.join(par_dir, 'tile'+str(t))
        # If this path doesn't already exist, make the path
        if not os.path.exists(tile_dir):
            os.mkdir(tile_dir)
        # Update the value in the file_names section
        cfg.set(section='file_names', option='output_dir', value=tile_dir)

        # Now update the basic_info, change use_tiles to tile t
        cfg.set(section='basic_info', option='use_tiles', value=str(t))

        # Now write this to a new config file
        new_config_file = os.path.join(tile_dir, 'config'+str(t)+'.ini')
        with open(new_config_file, 'w') as file_path:
            cfg.write(file_path)

        config_file_path.append(new_config_file)

    return config_file_path

