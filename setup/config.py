# Load config files for the Python port of the ISS pipeline.

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
import os.path
import re

# List of options and their type.  If you change this, update the
# config.default.ini file too.  Make sure the type is valid.
_options = {
    'basic_info':
        {
            '3d': 'bool',
            'anchor_channel': 'maybe_int',
            'dapi_channel': 'maybe_int',
            'ref_round': 'int',
            'ref_channel': 'int',
            'use_channels': 'maybe_list_int',
            'use_rounds': 'maybe_list_int',
            'use_z': 'maybe_list_int',
            'use_tiles': 'maybe_list_int',
            'tile_pixel_value_shift': 'int',
            'ignore_first_z_plane': 'bool'
        },
    'file_names':
        {
            'input_dir': 'dir',
            'output_dir': 'dir',
            'tile_dir': 'dir',
            'round': 'list',
            'anchor': 'maybe_str',
            'raw_extension': 'str'
        },
    'extract':
        {
            'wait_time': 'int',
            'r1': 'maybe_int',
            'r2': 'maybe_int',
            'r_dapi': 'maybe_int',
            'r1_auto_microns': 'number',
            'r_dapi_auto_microns': 'number',
            'scale': 'maybe_number',
            'scale_norm': 'maybe_int',
            'scale_tile': 'maybe_int',
            'scale_channel': 'maybe_int',
            'scale_z': 'maybe_int',
            'scale_anchor': 'maybe_number',
            'scale_anchor_tile': 'maybe_int',
            'scale_anchor_z': 'maybe_int'
        },
    'find_spots':
        {
            'min_score': 'maybe_number',
            'step': 'list'
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
    'str': lambda x: True,
    'bool': lambda x: re.match("True|true|False|false", x) is not None,
    'file': lambda x: os.path.isfile(x),
    'dir': lambda x: os.path.isdir(x),
    'list': lambda x: True,
    'list_int': lambda x: all([_option_type_checkers['int'](s.strip()) for s in x.split(",")]),
    'list_number': lambda x: all([_option_type_checkers['number'](s.strip()) for s in x.split(",")]),
    'maybe_int': lambda x: x.strip() == "" or _option_type_checkers['int'](x),
    'maybe_number': lambda x: x.strip() == "" or _option_type_checkers['number'](x),
    'maybe_list_int': lambda x: x.strip() == "" or _option_type_checkers['list_int'](x),
    'maybe_str': lambda x: x.strip() == "" or _option_type_checkers['str'](x)
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
    'maybe_int': lambda x: None if x == "" else _option_formatters['int'](x),
    'maybe_number': lambda x: None if x == "" else _option_formatters['number'](x),
    'maybe_list_int': lambda x: None if x == "" else _option_formatters['list_int'](x),
    'maybe_str': lambda x: None if x == "" else _option_formatters['str'](x)
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
    with open(ini_file, 'r') as f:
        _parser.read_string(f.read())

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
            # 2. Ensure there are no extra options in the config file.
            if name not in _options[section].keys():
                raise InvalidConfigError(section, name, val)
            # 3. Ensure that all of the option values pass type checking.
            if not _option_type_checkers[_options[section][name]](val):
                raise InvalidConfigError(section, name, val)

    # Now that we have validated, build the configuration dictionary
    out_dict = {section: {} for section in _options.keys()}
    for section in _options.keys():
        for name, val in _parser[section].items():
            out_dict[section][name] = _option_formatters[_options[section][name]](_parser[section][name])
    return out_dict

# Config = get_config("settings.ini")
