from .. import config
import tempfile
import unittest
import os


class TestConfig(unittest.TestCase):
    DEFAULT_CONFIG_FILE = "setup/settings.default.ini"

    def setUp(self):
        # For each type, we have a tuple.  The first are strings which should
        # be interpreted as valid types, and the second is the values they
        # should be interpreted as in Python.
        self.examples = {"int": (["0", "100"],
                                 [0, 100]),
                         "number": (["0", "0.0", "3.141", "1"],
                                    [0, 0, 3.141, 1]),
                         "str": (["a", " ", "hello world"],
                                 ["a", " ", "hello world"]),
                         "bool": (["True", "true", "False", "false"],
                                  [True, True, False, False]),
                         "file": ([__file__],  # Easiest way to get any valid file I think?
                                  [__file__]),
                         "dir": ([os.getcwd()],  # Easiest way to get any valid directory I think?
                                 [os.getcwd()]),
                         "list": (["a,b,c", "", "a , b , c  ", "1, 2,3"],
                                  [["a", "b", "c"], [""], ["a", "b", "c"], ["1", "2", "3"]]),
                         "list_int": (["1,2,3", "  1  , 2 ,3"],
                                      [[1, 2, 3], [1, 2, 3]]),
                         "list_number": (["1,2.3,4.5"],
                                         [[1, 2.3, 4.5]]),
                         "list_str": (["a,b,c", "hello world", "a , b , c  ", "1, 2,3"],
                                      [["a", "b", "c"], ["hello world"], ["a", "b", "c"], ["1", "2", "3"]]),
                         "maybe_int": (["0", "100", ""],
                                       [0, 100, None]),
                         "maybe_number": (["0", "0.0", "3.141", "1", ""],
                                          [0, 0, 3.141, 1, None]),
                         "maybe_list_int": (["1,2,3", ""],
                                            [[1, 2, 3], None]),
                         "maybe_str": (["0", "hello", ""],
                                       ["0", "hello", None]),
                         "maybe_list_str": (["a,b,c", "", "a , b , c  ", "1, 2,3"],
                                            [["a", "b", "c"], None, ["a", "b", "c"], ["1", "2", "3"]]),
                         "maybe_list_number": (["1,2.3,4.5", ""],
                                               [[1, 2.3, 4.5], None]),
                         "maybe_file": ([__file__, ""],  # Easiest way to get any valid file I think?
                                        [__file__, None]),
                         }
        # Emtpy config file
        self.fake_configs = [self._make_fake_config(0), self._make_fake_config(-1)]

    def test_type_checkers_matches_formatters(self):
        self.assertTrue(list(sorted(config._option_type_checkers.keys())) == \
                        list(sorted(config._option_formatters.keys())),
                        "All type checkers must have an associated formatter")

    def _make_fake_config(self, index):
        # Construct a fake configuration file based on the options in config
        config_text = ""
        expected_config = {}
        for section, items in config._options.items():
            config_text += f"[{section}]\n"
            for k, typ in items.items():
                config_text += f"{k} = {self.examples[typ][0][index]}\n"
                expected_config[k] = self.examples[typ][1][index]
        return config_text

    def test_config_file_or_string(self):
        # First try with an empty configuration
        for fake_config in self.fake_configs:
            fake_config_string = config.get_config(fake_config)
            with tempfile.TemporaryDirectory() as d:
                fn = os.path.join(d, "configtest.ini")
                with open(fn, "w") as f:
                    f.write(fake_config)
                fake_config_file = config.get_config(fn)
            for k in fake_config_file.keys():
                for k2 in fake_config_file[k]:
                    self.assertEqual(fake_config_file[k][k2], fake_config_string[k][k2])
            for k in fake_config_string.keys():
                for k2 in fake_config_string[k]:
                    self.assertEqual(fake_config_file[k][k2], fake_config_string[k][k2])

    def test_all_valid_types_have_unit_tests(self):
        self.assertTrue(list(sorted(self.examples.keys())) == list(sorted(config._option_type_checkers.keys())),
                        "Make sure to create unit tests for all new types")

    def test_types_equal_expected_values(self):
        for k in self.examples.keys():
            for example, actual in zip(*self.examples[k]):
                self.assertEqual(config._option_formatters[k](example), actual)

    def test_types_pass_validation(self):
        for k in self.examples.keys():
            for example in self.examples[k][0]:
                self.assertTrue(config._option_type_checkers[k](example),
                                f"Error with {k}: {example!r} doesn't type check")
