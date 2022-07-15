from .. import notebook
import tempfile
import unittest
import os
import numpy as np
import json


class TestNotebook(unittest.TestCase):
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/simplest_settings.ini')

    def test_create_Notebook(self):
        with tempfile.TemporaryDirectory() as d:
            # Test filenames with no extension, a "bad" extension, and the correct
            # "npz" extension
            for fn in ["nbfile1", "nbfile2.ext", "nbfile3.npz"]:
                nb = notebook.Notebook(os.path.join(d, fn), self.CONFIG_FILE)

    def test_create_NotebookPage(self):
        nbp = notebook.NotebookPage("name")

    def test_add_NotebookPage_to_Notebook(self):
        with tempfile.TemporaryDirectory() as d:
            # Test using standard syntax
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nbp.val = 2
            self.assertEqual(len(nb), 0)
            nb.add_page(nbp)
            self.assertEqual(len(nb), 1)
            self.assertEqual(nb.pagename.val, 2)
            # Test using syntactic sugar
            nb2 = notebook.Notebook(os.path.join(d, "file2"), self.CONFIG_FILE)
            nbp2 = notebook.NotebookPage("pagename")
            nbp2.val = 2
            self.assertEqual(len(nb2), 0)
            nb2 += nbp2
            self.assertEqual(len(nb2), 1)
            self.assertEqual(nb2.pagename.val, 2)

    def test_save_Notebook(self):
        with tempfile.TemporaryDirectory() as d:
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nbp.item = 3
            nb += nbp  # This triggers the save
            nb_reload = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            self.assertEqual(nb, nb_reload)

    def test_Notebook_version_hash(self):
        with tempfile.TemporaryDirectory() as d:
            # A baseline notebook
            nb = notebook.Notebook(os.path.join(d, "file1"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nbp.item = 3
            nb += nbp
            # The reference should have the same hash as itself
            nb_reloaded = notebook.Notebook(os.path.join(d, "file1"), self.CONFIG_FILE)
            self.assertEqual(nb_reloaded.version_hash(), nb.version_hash())
            # The reference should have the same hash as this one, even though they
            # are unequal
            nb2 = notebook.Notebook(os.path.join(d, "file2"), self.CONFIG_FILE)
            nbp2 = notebook.NotebookPage("pagename")
            nbp2.item = 5
            nb2 += nbp2
            self.assertNotEqual(nb, nb2)
            self.assertEqual(nb.version_hash(), nb2.version_hash())
            # The reference should have a different hash than this one.
            nb3 = notebook.Notebook(os.path.join(d, "file3"), self.CONFIG_FILE)
            nbp3 = notebook.NotebookPage("pagename")
            nbp3.item2 = 5
            nb3 += nbp3
            self.assertNotEqual(nb.version_hash(), nb3.version_hash())

    def test_equal_with_nan(self):
        # test equals functionality works if there are nans in the data
        nbp = notebook.NotebookPage("pagename")
        nbp.nan_array = np.ones((5, 10)) * np.nan
        self.assertEqual(nbp, nbp)
        nbp2 = notebook.NotebookPage("page2")
        nbp2.nan_array = np.zeros((8, 12))
        nbp2.nan_array[5, 3] = np.nan
        nbp2.nan_array[1, 11] = np.nan
        nbp2.nan_array[0, 0] = np.nan
        self.assertEqual(nbp2, nbp2)
        nbp3 = notebook.NotebookPage("page3")
        nbp3.nan_val = np.nan
        self.assertEqual(nbp3, nbp3)

    def test_load_diff_config(self):
        config_file2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples/simplest_settings2.ini')
        with tempfile.TemporaryDirectory() as d:
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nb += notebook.NotebookPage("pagename")
            nb_reloaded = notebook.Notebook(os.path.join(d, "file"), config_file2)
            self.assertNotEqual(nb_reloaded, nb)  # Different because config are different
            self.assertNotEqual(nb_reloaded._config, nb._config)  # Different because configs are different
            # True because configs only different in section for which page not added and in file_names section
            # which can be different.
            self.assertTrue(nb.compare_config(nb_reloaded.get_config()))

    @unittest.expectedFailure
    def test_NotebookPage_writeonce(self):
        nbp = notebook.NotebookPage("pagename")
        nbp.x = 3
        nbp.x = 2

    @unittest.expectedFailure
    def test_Notebook_writeonce(self):
        with tempfile.TemporaryDirectory() as d:
            # A baseline notebook
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nb += nbp
            nb += nbp

    @unittest.expectedFailure
    def test_Notebook_same_name_page(self):
        with tempfile.TemporaryDirectory() as d:
            # A baseline notebook
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nbp2 = notebook.NotebookPage("pagename")
            nb += nbp
            nb += nbp2

    @unittest.expectedFailure
    def test_NotebookPage_readonly_after_added_to_Notebook(self):
        with tempfile.TemporaryDirectory() as d:
            # A baseline notebook
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nb += nbp
            nb.pagename.newvar = 1

    @unittest.expectedFailure
    def test_Notebook_no_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            # A baseline notebook
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nb.newpage = nbp  # "newpage" is not equal to nbp.name ("pagename") hence it fails

    @unittest.expectedFailure
    def test_NotebookPage_add_var_not_in_comments(self):
        with open(notebook.NotebookPage._comments_file) as f:
            json_comments = json.load(f)
        page_name = list(json_comments.keys())[0]
        nbp = notebook.NotebookPage(page_name)
        nbp.RFhzdfuW3532 = 22

    @unittest.expectedFailure
    def test_NotebookPage_add_var_called_DESCRIPTION(self):
        with open(notebook.NotebookPage._comments_file) as f:
            json_comments = json.load(f)
        page_name = list(json_comments.keys())[0]
        nbp = notebook.NotebookPage(page_name)
        nbp.DESCRIPTION = 22

    @unittest.expectedFailure
    def test_Notebook_add_page_not_all_comment_values(self):
        with tempfile.TemporaryDirectory() as d:
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            with open(notebook.NotebookPage._comments_file) as f:
                json_comments = json.load(f)
            page_name = list(json_comments.keys())[0]
            nbp = notebook.NotebookPage(page_name)
            nb += nbp

    @unittest.expectedFailure
    def test_Notebook_add_page_extra_value(self):
        with tempfile.TemporaryDirectory() as d:
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            with open(notebook.NotebookPage._comments_file) as f:
                json_comments = json.load(f)
            page_name = list(json_comments.keys())[0]
            nbp = notebook.NotebookPage('blah')
            nbp.dnc67e2 = 5  # extra value not in comments file
            for key in json_comments[page_name]:
                if key != "DESCRIPTION":
                    nbp.__setattr__(key, 5)  # add all values in comments file
            nbp.name = page_name  # change name to that of section in comments file
            nb += nbp  # should fail due to extra value not in comments file

    def test_types(self):
        # Update this test when new types are added
        with tempfile.TemporaryDirectory() as d:
            nb = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            nbp = notebook.NotebookPage("pagename")
            nbp.str_type = "hello world"
            nbp.int_type = 3
            nbp.float_type = 4.1
            nbp.array_type = np.asarray([[1, 2], [3, 4]])
            nbp.list_string = ["hi", "hi2", "hi3"]
            nbp.list_int = [1, 5, 2]
            nbp.list_number = [0.32553, 0.9003, 65.1]
            nbp.list_mixed = ["hi", 0.34, 5, 100, 12.3456]
            nbp.list_2d = [[5, 2], [3, 4], [1, 2]]
            nbp.list_3d = np.random.random((5, 5, 4)).tolist()
            nbp.bool1 = True
            nbp.bool2 = False
            nbp.none = None
            nbp.none_string = "None"
            nbp.nan = np.nan
            nb += nbp  # Triggers save
            nb_reloaded = notebook.Notebook(os.path.join(d, "file"), self.CONFIG_FILE)
            self.assertEqual(nb_reloaded, nb)


if __name__ == '__main__':
    unittest.main()
