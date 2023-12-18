"""The Notebook is a write-once data structure which saves the output of
various stages of the neuromics pipeline.  Each "page" of the Notebook is
itself a write-once data structure.  Each page may contain many different
entries.  To use a notebook, first create an empty notebook, associated with a
file.  Then, in a subroutine, create a NotebookPage object. All entries must be
added to a NotebookPage, not directly to the notebook.  The NotebookPage must
have a descriptive name describing it.  Usually, this name should be the stage
in the pipeline for which the NotebookPage contains results.  Whenever an entry
is added to a NotebookPage, in addition to saving the value, it saves the time
at which the entry was added.  Likewise, the time at which a NotebookPage is
created, and the time at which it is added to the lab book, are also recorded
automatically.  This both serves as a record of what was done, as well as a
source for debugging and optimization.

Conceptually, the idea is that a Notebook is like a lab notebook.  In a lab
notebook, you write things in a separate section (here, page) for each part of
the experiment with the appropriate section name.  You only add, you never
erase or modify.  Lab notebooks contain intermediate results, as well as the
main data collected during the experiment.  All times and labels of all results
are written down.

One important implementation detail: we automatically assign a type to each
entry.  The purpose of this type is exclusively to determine the procedure for
saving and loading.  Since we save the Notebook as an npz file, and npz files
can only consist of numpy objects, this system provides a strategy for
converting non-numpy objects to and from numpy objects.  This is in contrast to
the types for the config file, which are designed for data validation.  Here,
the user does not need to think about types, which should be used seamlessly
and silently in the background.  (If a new type is needed, see the
documentation near the TYPES variable in the code below for how to add a new
type.)

"""
import numpy as np
import hashlib
import os
import time
import json
import warnings
import copy
from pathlib import PurePath
try:
    import importlib_resources
except ModuleNotFoundError:
    import importlib.resources as importlib_resources
from typing import Tuple, List, Optional

from .config import get_config
from .file_names import set_file_names


# Functions in Notebook._no_save_pages need defined here

def load_file_names(nb, page_name: str):
    # bridge function to avoid circular import
    return set_file_names(nb, NotebookPage(page_name))


# The variable TYPES defines strategies for saving and loading different kinds
# of variables.  Each type is defined by a length-three tuple: first is the
# name, a string which is used to reference it.  Second is a function to test
# whether a variable is a given type, returning True if it is and False if it
# is not.  Third is a function to convert a value from a saved npz file back to
# the Python equivalent.  The name of the type is saved inside the npz file, so
# the proper type is sure to be loaded.
#
# Also, note that the ordering of the TYPES list is important Types are tested
# sequentially, starting with the first, and continuing iteratively until a
# valid type si found.  In other words, if a variables satisfies multiple
# types, the assigned type will be the first matching type in the TYPES list.
#
# Finally, if you create a new type, please add a unit test for it.
TYPES = [
    ("boolean",  # needs to be first as isinstance(True, int) is True
     lambda x: isinstance(x, bool),
     lambda x: bool(x[()]),
     ),
    ("string",
     lambda x: isinstance(x, (str, np.str_)),
     lambda x: str(x[()]),
     ),
    ("list",
     lambda x: isinstance(x, list),
     lambda x: list(x),
     ),
    ("none",  # saved in ndz file as 'None'
     lambda x: x is None,
     lambda x: None,
     ),
    ("ndarray",
     lambda x: isinstance(x, np.ndarray),
     lambda x: x,
     ),
    ("int",
     lambda x: isinstance(x, (int, np.int_)),
     lambda x: int(x[()]),
     ),
    ("number",
     lambda x: isinstance(x, (float, np.float_)) and np.isreal(x) is True,
     lambda x: float(x[()]),
     ),
]


def _decode_type(key, val, typ):
    """Convert a value from an npz file to a Python value.

    The value saved in the npz file, `val`, is converted to a compatible Python
    variable.  It is converted as if it is the type `typ`, which must be saved
    alongside it in the npz file.  The name of the value should be given as
    `key`.  (We don't actually need `key`, but it helps us provide useful error
    messages.)
    """
    for n, _, f in TYPES:
        if n == typ:
            return f(val)
    raise TypeError(f"Key {key!r} has type {typ!r}, "
                    "but we don't know how to decode that.  "
                    f"Please use one of the following: {[t[0] for t in TYPES]}")


def _get_type(key, val):
    """Find the type of a given value.

    We don't know how to save all types of variables.  This function checks to
    make sure we know how to save the given variable.  If we do know how to
    save it, it returns the type it can be saved as.  `key` is the name of the
    entry and `val` is the value to check the type of.  (We don't actually need
    `key` but it helps us provide useful error messages.)

    Type is checked using the following procedure.  It steps through the
    elements of TYPES one by one.  Each element of TYPES should be a tuple,
    where the first element is the name of the type and the second element is a
    function that tests whether an element is a part of the type (as described
    above).  Note that order matters here: if two functions return True, the
    first one in TYPES will be used as the type.
    """
    for n, f, _ in TYPES:
        if f(val):
            return n
    raise TypeError(f"Key {key!r} has value {val!r} which "
                    f"is of type {type(val)}, which is invalid.  "
                    f"Please use one of the following: {[t[0] for t in TYPES]}")


# Standard formatting for errors in the config file
class InvalidNotebookPageError(Exception):
    """Exception for an invalid notebook page item"""

    def __init__(self, page_var_name, comments_var_name, page_name):
        if comments_var_name is None:
            if page_var_name == "DESCRIPTION":
                error = f"Cannot assign {page_var_name} because in comments file, " \
                        f"this key is used to describe whole page."
            else:
                error = f"Cannot assign {page_var_name} because it is not in comments file for the {page_name} page."
        else:
            if page_var_name is None:
                error = f"Cannot add {page_name} page to notebook because the key {comments_var_name} in the " \
                        f"comments page does not have a value in the page."
            else:
                error = f"No variables provided to give error comment"
        super().__init__(error)


class Notebook:
    """
    Notebook
    ========
    A write-only file-synchronized class to keep track of *coppaFISH* results.

    The `Notebook` object stores all of the outputs of the script.  Almost all
    information saved in the `Notebook` is encapsulated within `"pages"`, from the
    `NotebookPage` object.  To add a `NotebookPage` object to a `Notebook`, use the
    `"add_page"` method.
    In addition to saving pages, it also saves the contents of the
    config file, and the time at which the notebook and each page was created.

    Usage:
    ------
    To create a `Notebook`, pass it the path to the file where the `Notebook` is to
    be stored (`notebook_file`), and optionally, the path to the configuration file
    (`config_file`).  If `notebook_file` already exists, the notebook located
    at this path will be loaded.  If not, a new file will be created as soon as
    the first data is written to the `Notebook`.

    Example:
    --------
        === "With config_file"

            ``` python
            nb = Notebook("nbfile.npz", "config_file.ini")
            nbp = NotebookPage("pagename")
            nbp.var = 1
            nb.add_page(nbp) or nb += nbp or nb.pagename = nbp
            assert nb.pagename.var == 1
            ```

        === "No config_file"

            ``` python
            nb = Notebook("nbfile.npz")
            nbp = NotebookPage("pagename")
            nbp.var = 1
            nb.add_page(nbp) or nb += nbp or nb.pagename = nbp
            assert nb.pagename.var == 1
            ```

    Because it is automatically saved to the disk, you can close Python, reopen
    it, and do the following (Once `config_file`, added to notebook there is no need to load it again unless it has
    been changed):
    ```python
    nb2 = Notebook("nbfile.npz")
    assert nb2.pagename.var == 1
    ```

    If you create a notebook without specifying `notebook_file`, i.e.
    ```nb = Notebook(config_file="config_file.ini")```, the `notebook_file` will be set to:
    ```python
    notebook_file = config['file_names']['output_dir'] + config['file_names']['notebook_name']).
    ```.

    Notes:    
        - "On using config_file", when running the coppafish pipeline, the `Notebook` requires a `config_file` to 
        access information required for the different stages of the pipeline through `nb.get_config()`. But if using 
        the `Notebook` to store information not in coppafish pipeline, it is not needed.
    """
    _SEP = "_-_"  # Separator between notebook page name and item name when saving to file
    _ADDEDMETA = "TIME_CREATED"  # Key for notebook created time
    _CONFIGMETA = "CONFIGFILE"  # Key for config string
    _NBMETA = "NOTEBOOKMETA"  # Key for metadata about the entire notebook
    # If these sections of config files are different, will not raise error.
    _no_compare_config_sections = ['file_names']

    # When the pages corresponding to the keys are added, a save will not be triggered.
    # When save does happen, these pages won't be saved, but made on loading using
    # the corresponding function, load_func, if the notebook contains the pages indicated by
    # load_func_req.
    # load_func must only take notebook and page_name as input and has no output but page will be added to notebook.
    # When last of pages in load_func_req have been added, the page will automatically be added.
    _no_save_pages = {'file_names': {'load_func': load_file_names, 'load_func_req': ['basic_info']}}

    def __init__(self, notebook_file: Optional[str] = None, config_file: Optional[str] = None):
        # Give option to load with config_file as None so don't have to supply ini_file location every time if
        # already initialised.
        # Also, can provide config_file if file_names section changed.
        # Don't need to provide notebook_file as can determine this from config_file as:
        # config['file_names']['output_dir'] + config['file_names']['notebook_name']

        # numpy isn't compatible with npz files which do not end in the suffix
        # .npz.  If one isn't there, it will add the extension automatically.
        # We do the same thing here.
        object.__setattr__(self, '_page_times', {})
        if notebook_file is None:
            if config_file is None:
                raise ValueError('Both notebook_file and config_file are None')
            else:
                config_file_names = get_config(config_file)['file_names']
                notebook_file = os.path.join(config_file_names['output_dir'], config_file_names['notebook_name'])
                if not os.path.isdir(config_file_names['output_dir']):
                    raise ValueError(f"\nconfig['file_names']['output_dir'] = {config_file_names['output_dir']}\n"
                                     f"is not a valid directory.")
        if not notebook_file.endswith(".npz"):
            notebook_file = notebook_file + ".npz"
        # Note that the ordering of _pages may change across saves and loads,
        # but the order will always correspond to the order of _pages_times
        self._file = notebook_file
        self._config_file = config_file
        # Read the config file, but don't assign anything yet.  Here, we just
        # save a copy of the config file.  This isn't the main place the config
        # file should be read from.
        if config_file is not None:
            if os.path.isfile(str(config_file)):
                with open(config_file, 'r') as f:
                    read_config = f.read()
            else:
                raise ValueError(f'Config file given is not valid: {config_file}')
        else:
            read_config = None
        # If the file already exists, initialize the Notebook object from this
        # file.  Otherwise, initialize it empty.
        if os.path.isfile(self._file):
            pages, self._page_times, self._created_time, self._config = self.from_file(self._file)
            for page in pages:
                object.__setattr__(self, page.name, page)  # don't want to set page_time hence use object setattr
            if read_config is not None:
                if not self.compare_config(get_config(read_config)):
                    raise SystemError("Passed config file is not the same as the saved config file")
                self._config = read_config  # update config to new one - only difference will be in file_names section
            self.add_no_save_pages()  # add file_names page with new config
        else:
            print("Notebook file not found, creating a new notebook.")
            if read_config is None:
                warnings.warn("Have not passed a config_file so Notebook.get_config() won't work.")
            self._created_time = time.time()
            self._config = read_config

    def __repr__(self):
        # This means that print(nb) gives file location of notebook and
        # pages in the notebook sorted by time added to the notebook.
        sort_page_names = sorted(self._page_times.items(), key=lambda x: x[1])  # sort by time added to notebook
        page_names = [name[0] for name in sort_page_names]
        n_names_per_line = 4
        i = n_names_per_line - 1
        while i < len(page_names) - n_names_per_line / 2:
            page_names[i + 1] = "\n" + page_names[i + 1]
            i = i + n_names_per_line
        page_names = ", ".join(page_names)
        return f"File: {self._file}\nPages: {page_names}"

    def get_config(self):
        """
        Returns config as dictionary.
        """
        if self._config is not None:
            return get_config(self._config)
        else:
            raise ValueError('Notebook does not contain config parameter.')

    def compare_config(self, config_2: dict) -> bool:
        """
        Compares whether `config_2` is equal to the config file saved in the notebook.
        Only sections not in `_no_compare_config_sections` and with a corresponding page saved to the notebook
        will be checked.

        Args:
            config_2: Dictionary with keys corresponding to sections where a section
                is also a dictionary containing parameters.
                E.g. `config_2['basic_info]['param1'] = 5`.

        Returns:
            `True` if config dictionaries are equal in required sections.

        """
        # TODO: issue here that if default settings file changed, the equality here would still be true.
        config = self.get_config()
        is_equal = True
        if config.keys() != config_2.keys():
            warnings.warn('The config files have different sections.')
            is_equal = False
        else:
            sort_page_names = sorted(self._page_times.items(), key=lambda x: x[1])  # sort by time added to notebook
            # page names are either same as config sections or with _debug suffix
            page_names = [name[0].replace('_debug', '') for name in sort_page_names]
            for section in config.keys():
                # Only compare sections for which there is a corresponding page in the notebook.
                if section not in self._no_compare_config_sections and section in page_names:
                    if config[section] != config_2[section]:
                        warnings.warn(f"The {section} section of the two config files differ.")
                        is_equal = False
        return is_equal

    def describe(self, key=None):
        """
        `describe(var)` will print comments for variables called `var` in each `NotebookPage`.
        """
        if key is None:
            print(self.__repr__())
        elif len(self._page_times) == 0:
            print(f"No pages so cannot search for variable {key}")
        else:
            sort_page_names = sorted(self._page_times.items(), key=lambda x: x[1])  # sort by time added to notebook
            page_names = [name[0] for name in sort_page_names]
            first_page = self.__getattribute__(page_names[0])
            with open(first_page._comments_file) as f:
                json_comments = json.load(f)
            if self._config is not None:
                config = self.get_config()
            n_times_appeared = 0
            for page_name in page_names:
                # if in comments file, then print the comment
                if key in json_comments[page_name]:
                    print(f"{key} in {page_name}:")
                    self.__getattribute__(page_name).describe(key)
                    print("")
                    n_times_appeared += 1

                elif self._config is not None:
                    # if in config file, then print the comment
                    # find sections in config file with matching name to current page
                    config_sections_with_name = [page_name.find(list(config.keys())[i]) for i in
                                                 range(len(config.keys()))]
                    config_sections = np.array(list(config.keys()))[np.array(config_sections_with_name) != -1]
                    for section in config_sections:
                        for param in config[section].keys():
                            if param.lower() == key.lower():
                                print(f"No variable named {key} in the {page_name} page.\n"
                                      f"But it is in the {section} section of the config file and has value:\n"
                                      f"{config[section][param]}\n")
                                n_times_appeared += 1
            if n_times_appeared == 0:
                print(f"{key} is not in any of the pages in this notebook.")

    def __eq__(self, other):
        # Test if two `Notebooks` are identical
        #
        # For two `Notebooks` to be identical, all aspects must be the same,
        # excluding the ordering of the pages, and the filename.  All timestamps
        # must also be identical.

        if self._created_time != other._created_time:
            return False
        if self._config != other._config:
            return False
        if len(self._page_times) != len(other._page_times):
            return False
        for k in self._page_times.keys():
            if k not in other._page_times or getattr(self, k) != getattr(other, k):
                return False
        for k in other._page_times.keys():
            if k not in self._page_times or getattr(other, k) != getattr(self, k):
                return False
        for k, v in self._page_times.items():
            if k not in other._page_times or v != other._page_times[k]:
                return False
        return True

    def __len__(self):
        # Return the number of pages in the Notebook
        return len(self._page_times)

    def __setattr__(self, key, value):
        # Deals with the syntax `nb.key = value`
        # automatically triggers save if `NotebookPage` is added.
        # If adding something other than a `NotebookPage`, this syntax does exactly as it is for other classes.
        if isinstance(value, NotebookPage):
            if self._SEP in key:
                raise NameError(f"The separator {self._SEP} may not be in the page's name")
            if value.finalized:
                raise ValueError("Page already added to a Notebook, cannot add twice")
            if key in self._page_times.keys():
                raise ValueError("Cannot add two pages with the same name")
            if value.name != key:
                raise ValueError(f"Page name is {value.name} but key given is {key}")

            # ensure all the variables in the comments file are included
            with open(value._comments_file) as f:
                json_comments = json.load(f)
            if value.name in json_comments:
                for var in json_comments[value.name]:
                    if var not in value._times and var != "DESCRIPTION":
                        raise InvalidNotebookPageError(None, var, value.name)
                # ensure all variables in page are in comments file
                for var in value._times:
                    if var not in json_comments[value.name]:
                        raise InvalidNotebookPageError(var, None, value.name)

            value.finalized = True
            object.__setattr__(self, key, value)
            self._page_times[key] = time.time()
            if value.name not in self._no_save_pages.keys():
                self.save()
            self.add_no_save_pages()
        elif key in self._page_times.keys():
            raise ValueError(f"Page with name {key} in notebook so can't add variable with this name.")
        else:
            object.__setattr__(self, key, value)

    def __delattr__(self, name):
        # Method to delete a page or attribute. Deals with del nb.name.
        object.__delattr__(self, name)
        if name in self._page_times:
            # extra bit if page
            del self._page_times[name]

    def add_page(self, page):
        """Insert the page `page` into the `Notebook`.

        This function automatically triggers a save.
        """
        if not isinstance(page, NotebookPage):
            raise ValueError("Only NotebookPage objects may be added to a notebook.")
        self.__setattr__(page.name, page)

    def has_page(self, page_name):
        """A check to see if notebook includes a page called page_name.
        If page_name is a list, a boolean list of equal size will be
        returned indicating whether each page is present."""
        if isinstance(page_name, str):
            output = any(page_name == p for p in self._page_times)
        elif isinstance(page_name, list):
            output = [any(page_name[i] == p for p in self._page_times) for i in range(len(page_name))]
        else:
            raise ValueError(f"page_name given was {page_name}. This is not a list or a string.")
        return output

    def __iadd__(self, other):
        # Syntactic sugar for the add_page method
        self.add_page(other)
        return self

    def add_no_save_pages(self):
        """
        This adds the page `page_name` listed in `nb._no_save_pages` to the notebook if
        the notebook already contains the pages listed in `nb._no_save_pages['page_name']['load_func_req']`
        by running the function `nb._no_save_pages['page_name']['load_func'](nb, 'page_name')`.

        At the moment, this is only used to add the `file_names` page to the notebook as soon as the `basic_info` page
        has been added.
        """
        for page_name in self._no_save_pages.keys():
            if self.has_page(page_name):
                continue
            if all(self.has_page(self._no_save_pages[page_name]['load_func_req'])):
                # If contains all required pages to run load_func, then add the page
                self._no_save_pages[page_name]['load_func'](self, page_name)

    def change_page_name(self, old_name: str, new_name: str):
        """
        This changes the name of the page `old_name` to `new_name`. It will trigger two saves,
        one after changing the new and one after changing the time the page was added to be the time
        the initial page was added.

        Args:
            old_name:
            new_name:
        """
        nbp = self.__getattribute__(old_name)
        warnings.warn(f"Changing name of {old_name} page to {new_name}")
        time_added = self._page_times[old_name]
        nbp.finalized = False
        nbp.name = new_name
        self.__delattr__(old_name)
        self.add_page(nbp)
        self._page_times[new_name] = time_added  # set time to time page initially added
        self.save()

    def version_hash(self):
        # A short string representing the file version.
        #
        # Since there are many possible page names and entry names within those
        # pages, that means there are many, many possible file versions based on
        # different versions of the code.  Rather than try to keep track of these
        # versions and appropriately increment some centralized counter, we
        # generate a short string which is a hash of the page names and the names
        # of the entries in that page.  This way, it is possible to see if two
        # notebooks were generated using the same version of the software.  (Of
        # course, it assumes that no fields are ever set conditionally.)

        s = ""
        for p_name in self._page_times:
            s += p_name + "\n\n"
            page = getattr(self, p_name)
            s += "\n".join(sorted(page._times.keys()))
        return hashlib.md5(bytes(s, "utf8")).hexdigest()

    def save(self, file: Optional[str] = None):
        """
        Saves Notebook as a npz file at the path indicated by `file`.
        
        Args:
            file: Where to save *Notebook*. If `None`, will use `self._file`.

        """
        """Save the Notebook to a file"""
        if file is not None:
            if not file.endswith(".npz"):
                file = file + ".npz"
            self._file = file
        d = {}
        # Diagnostic information about how long the save took.  We can probably
        # take this out, or else set it at a higher debug level via warnings
        # module.
        save_start_time = time.time()
        for p_name in self._page_times.keys():
            if p_name in self._no_save_pages.keys():
                continue
            p = getattr(self, p_name)
            pd = p.to_serial_dict()
            for k, v in pd.items():
                if v is None:
                    # save None objects as string then convert back to None on loading
                    v = str(v)
                d[p_name + self._SEP + k] = v
            d[p_name + self._SEP + self._ADDEDMETA] = self._page_times[p_name]
        d[self._NBMETA + self._SEP + self._ADDEDMETA] = self._created_time
        if self._config is not None:
            d[self._NBMETA + self._SEP + self._CONFIGMETA] = self._config
        np.savez_compressed(self._file, **d)
        # Finishing the diagnostics described above
        print(f"Notebook saved: took {round(time.time() - save_start_time, 3)} seconds")
        if len(self.get_unique_versions()) > 1:
            warnings.warn(f"Saved notebook contains more than one software version: \n\t{self.get_unique_versions()}")

    def from_file(self, fn: str) -> Tuple[List, dict, float, str]:
        """
        Read a `Notebook` from a file

        Args:
            fn: Filename of the saved `Notebook` to load.

        Returns:
            A list of `NotebookPage` objects
            A dictionary of timestamps, of identical length to the list of `NotebookPage` objects and
                keys are `page.name`
            A timestamp for the time the `Notebook` was created.
            A string of the config file
        """
        # Right now we won't use lazy loading.  One problem with lazy loading
        # is that we must keep the file handle open.  We would rather not do
        # this, because if we write to the file, it will get screwed up, and if
        # there is a network issue, it will also mess things up.  I can't
        # imagine that loading the notebook will be a performance bottleneck,
        # but if it is, we can rethink this decision.  It should be pretty easy
        # to lazy load the pages, but eager load everything in the page.
        f = np.load(fn)
        keys = list(f.keys())
        page_items = {}
        page_times = {}
        created_time = None
        config_str = None  # If no config saved, will stay as None. Otherwise, will be the config in str form.
        for pk in keys:
            p, k = pk.split(self._SEP, 1)
            if p in self._no_save_pages.keys():
                # This is to deal with the legacy case from old code where a no_save_page has been saved.
                # If this is the case, don't load in this page.
                continue
            if p == self._NBMETA:
                if k == self._ADDEDMETA:
                    created_time = float(f[pk])
                    continue
                if k == self._CONFIGMETA:
                    config_str = str(f[pk])
                    continue
            if k == self._ADDEDMETA:
                page_times[p] = float(f[pk])
                continue
            if p not in page_items.keys():
                page_items[p] = {}
            page_items[p][k] = f[pk]
        pages = [NotebookPage.from_serial_dict(page_items[d]) for d in sorted(page_items.keys())]
        for page in pages:
            page.finalized = True  # if loading from file, then all pages are final
        assert len(pages) == len(page_times), "Invalid file, lengths don't match"
        assert created_time is not None, "Invalid file, invalid created date"
        return pages, page_times, created_time, config_str

    def get_unique_versions(self) -> List[str]:
        """
        Get every unique software version contained in all notebook pages.

        Returns:
            List[str]: list of unique software versions.
        """
        versions = []
        if self.has_page("basic_info"):
            versions.append(self.basic_info.software_version)
        if self.has_page("scale"):
            versions.append(self.scale.software_version)
        if self.has_page("extract"):
            versions.append(self.extract.software_version)
        if self.has_page("filter"):
            versions.append(self.filter.software_version)
        if self.has_page("find_spots"):
            versions.append(self.find_spots.software_version)
        if self.has_page("register"):
            versions.append(self.register.software_version)
        if self.has_page("stitch"):
            versions.append(self.stitch.software_version)
        if self.has_page("ref_spots"):
            versions.append(self.ref_spots.software_version)
        if self.has_page("call_spots"):
            versions.append(self.call_spots.software_version)
        if self.has_page("omp"):
            versions.append(self.omp.software_version)
        return list(set(versions))


class NotebookPage:
    """
    NotebookPage
    ============
    
    A page, to be added to a `Notebook` object.

    Usage:
    ------
    `NotebookPage` should be created at the beginning of a large step in the analysis pipeline.  The name of the page 
    should reflect its function, and it will be used as the indexing key when it is added to a Notebook.  The 
    `NotebookPage` should be created at the beginning of the step in the pipeline, because then the timestamp will be 
    more meaningful.  As results are computed, they should be added.  This will provide a timestamp for each of the 
    results as well.  Then, at the end, the pipeline step should return a `NotebookPage`, which can then be added to 
    the `Notebook`.

    Example:
    --------
        ```python
            nbp = NotebookPage("extract_and_filter")
            nbp.scale_factor = 10
            ...
            return nbp
        ```
    """
    _PAGEMETA = "PAGEINFO"  # Filename for metadata about the page
    _TIMEMETA = "___TIME"  # Filename suffix for timestamp information
    _TYPEMETA = "___TYPE"  # Filename suffix for type information
    _NON_RESULT_KEYS = ['name', 'finalized']
    _comments_file = importlib_resources.files('coppafish.setup').joinpath('notebook_comments.json')

    def __init__(self, name, input_dict=None):
        # Would like to store the git commit id in each notebook page to keep track of versions
        self.finalized = False  # Set to true when added to a Notebook
        self._times = {}
        self.name = name
        self._time_created = time.time()
        if isinstance(input_dict, dict):
            self.from_dict(input_dict)

    def __eq__(self, other):
        # Test for equality using the == syntax.
        # To be honest, I don't know why you would ever need this, but it is very
        # useful for writing unit tests, so here it is.
        if not isinstance(other, self.__class__):
            return False
        if self.name != other.name:
            return False
        if self._time_created != other._time_created:
            return False
        for k in self._times.keys():
            if k not in other._times or not np.array_equal(getattr(self, k), getattr(other, k)):
                # second condition in case failed first because of nan == nan is False.
                # Need first condition as well because equal_nan=True gives error for strings.
                if k not in other._times or not np.array_equal(getattr(self, k), getattr(other, k), equal_nan=True):
                    return False
        for k in other._times.keys():
            if k not in self._times or not np.array_equal(getattr(other, k), getattr(self, k)):
                # second condition in case failed first because of nan == nan is False.
                # Need first condition as well because equal_nan=True gives error for strings.
                if k not in self._times or not np.array_equal(getattr(other, k), getattr(self, k), equal_nan=True):
                    return False
        for k, v in self._times.items():
            if k not in other._times or v != other._times[k]:
                return False
        return True

    def __len__(self):
        # Return the number of results in the NotebookPage
        return len(self._times)

    def _is_result_key(self, key):
        # Whether key is a result variable or part of the metadata
        if key in self._NON_RESULT_KEYS or key[0] == '_':
            return False
        else:
            return True

    def __repr__(self):
        # This means that print(nbp) gives description of page if available or name and time created if not.
        json_comments = json.load(open(self._comments_file))
        if self.name in json_comments:
            return "\n".join(json_comments[self.name]['DESCRIPTION'])
        else:
            time_created = time.strftime('%d-%m-%Y- %H:%M:%S', time.localtime(self._time_created))
            return f"{self.name} page created at {time_created}"

    def describe(self, key: Optional[str] = None):
        """
        Prints a description of the variable indicated by `key`.

        Args:
            key: name of variable to describe that must be in `self._times.keys()`.
                If not specified, will describe the whole page.

        """
        if key is None:
            print(self.__repr__())  # describe whole page if no key given
        else:
            if key not in self._times.keys():
                print(f"No variable named {key} in the {self.name} page.")
            else:
                json_comments = json.load(open(self._comments_file))
                if self.name in json_comments:
                    # Remove empty lines
                    while '' in json_comments[self.name][key]: json_comments[self.name][key].remove('')
                    # replace below removes markdown code indicators
                    print("\n".join(json_comments[self.name][key]).replace('`', ''))
                else:
                    print(f"No comments available for page called {self.name}.")

    def __setattr__(self, key, value):
        # Add an item to the notebook page.
        #
        # For a `NotebookPage` object `nbp`, this handles the syntax `nbp.key = value`.
        # It checks the key and value for validity, and then adds them to the
        # notebook.  Specifically, it implements a write-once mechanism.
        if self._is_result_key(key):
            if self.finalized:
                raise ValueError("This NotebookPage has already been added to a Notebook, no more values can be added.")
            assert isinstance(key, str), f"NotebookPage key {key!r} must be a string, not {type(key)}"
            _get_type(key, value)
            if key in self.__dict__.keys():
                raise ValueError(f"Cannot assign {key} = {value!r} to the notebook page, key already exists")
            with open(self._comments_file) as f:
                json_comments = json.load(f)
            if self.name in json_comments:
                if key not in json_comments[self.name]:
                    raise InvalidNotebookPageError(key, None, self.name)
                if key == 'DESCRIPTION':
                    raise InvalidNotebookPageError(key, None, self.name)
            self._times[key] = time.time()
        object.__setattr__(self, key, value)

    def __delattr__(self, name):
        # Method to delete a result or attribute. Deals with del nbp.name.
        # Can only delete attribute if page has not been finalized.
        if self.finalized:
            raise ValueError("This NotebookPage has already been added to a Notebook, no values can be deleted.")
        object.__delattr__(self, name)
        if name in self._times:
            # extra bit if _is_result_key
            del self._times[name]

    def has_item(self, key):
        """Check to see whether page has attribute `key`"""
        return key in self._times.keys()

    def from_dict(self, d):
        """
        Adds all string keys of dictionary d to page.
        Keys whose value is None will be ignored.
        """
        for key, value in d.items():
            if isinstance(key, (str, np.str_)):
                if value is not None:
                    self.__setattr__(key, value)

    def to_serial_dict(self):
        """Convert to a dictionary which can be written to a file.

        In general, this function shouldn't need to be called other than within
        a `Notebook` object.
        """
        keys = {}
        keys[self._PAGEMETA] = self.name
        keys[self._PAGEMETA + self._TIMEMETA] = self._time_created
        for rn in self._times.keys():
            r = getattr(self, rn)
            keys[rn] = r
            keys[rn + self._TIMEMETA] = self._times[rn]
            keys[rn + self._TYPEMETA] = _get_type(rn, r)
        return keys

    @classmethod
    def from_serial_dict(cls, d):
        """Convert from a dictionary to a `NotebookPage` object

        In general, this function shouldn't need to be called other than within
        a `Notebook` object.
        """
        # Note that this method will need to be updated if you update the
        # constructor.
        name = str(d[cls._PAGEMETA][()])
        n = cls(name)
        n._time_created = float(d[cls._PAGEMETA + cls._TIMEMETA])
        # n.finalized = d[cls._FINALIZEDMETA]
        for k in d.keys():
            # If we've already dealt with the key, skip it.
            if k.startswith(cls._PAGEMETA): continue
            # Each key has an associated "time" and "type" key.  We deal with
            # the time and type keys separately when dealing with the main key.
            if k.endswith(cls._TIMEMETA): continue
            if k.endswith(cls._TYPEMETA): continue
            # Now that we have a real key, add it to the page.
            object.__setattr__(n, k, _decode_type(k, d[k], str(d[k + cls._TYPEMETA][()])))
            n._times[k] = float(d[k + cls._TIMEMETA])
        return n


def merge_notebooks(nb_list: List[Notebook], master_nb: Notebook) -> Notebook:
    """
    Function which merges `len(nb_list)` notebooks into one. On issues where all notebooks should have the same value, 
    for example, notebook.basic_info.anchor_channel, an error will be raised if these do not match.

    Args:
        nb_list (list of Notebooks): single-tile notebooks to be merged.
        master_nb (Notebook): master notebook initialized from the original un-split config file. Contains 
            'basic_info', 'file_names' and 'scale'. Can contain 'extract' and 'extract_debug' pages, these will then 
            not be changed by merging. Any pages that are not within this notebook are created by merging the pages 
            from the list of notebooks, if they all exist.

    Returns:
        master_nb: merged master notebook.
        
    Raises:
        AssertionError: merge of the given Notebook list is not possible based on incompatible values.
    """
    assert len(nb_list) > 1, "Require at least two notebooks to merge"
    assert master_nb.has_page('file_names'), "Master notebook must contain 'file_names' page"
    assert master_nb.has_page('basic_info'), "Master notebook must contain 'basic_info' page"
    assert not master_nb.has_page('find_spots'), "Master notebook cannot contain 'find_spots' page"
    for nb in nb_list:
        assert len(nb.basic_info.use_tiles) == 1, "Each notebook must only contain one tile"
    
    # Initialize the master notebook basic info page
    master_nbp_basic = master_nb.basic_info

    # First reorder the notebooks so that they are in ascending tile order
    nbp_tiles = [nb.basic_info.use_tiles[0] for nb in nb_list]
    nb_list = [x for _, x in sorted(zip(nbp_tiles, nb_list))]
    
    # Check the notebooks contain extract + extract_debug page
    has_extract_and_debug = all([bool(nb.has_page('extract')*nb.has_page('extract_debug')) for nb in nb_list])
    if not has_extract_and_debug:
        return master_nb
    # When all notebook pages have extract page we make a list of all of them, merge them and then add page to master
    nbp_extract_list = [nb.extract for nb in nb_list]
    master_nbp_extract = merge_extract(nbp_extract_list, master_nbp_basic)
    master_nb += master_nbp_extract
    master_nb.extract.finalized = True
    # Make a list of all of extract_debug, merge them and then add page to master
    nbp_extract_debug_list = [nb.extract_debug for nb in nb_list]
    master_nbp_extract_debug = merge_extract_debug(nbp_extract_debug_list, master_nbp_basic)
    master_nb += master_nbp_extract_debug
    master_nb.extract_debug.finalized = True
    
    # Check for filter and filter_debug pages
    has_filter_and_debug = all(
        [nb.has_page('filter') for nb in nb_list] + [nb.has_page('filter_debug') for nb in nb_list]
    )
    if not has_filter_and_debug:
        return master_nb
    nbp_filter_list = [nb.filter for nb in nb_list]
    master_nbp_filter = merge_filter(nbp_filter_list, master_nbp_basic)
    nbp_filter_debug_list = [nb.filter_debug for nb in nb_list]
    master_nbp_filter_debug = merge_filter_debug(nbp_filter_debug_list, master_nbp_basic)
    master_nb += master_nbp_filter
    master_nb.filter.finalized = True
    master_nb += master_nbp_filter_debug
    master_nb.filter_debug.finalized = True

    # Check the notebooks contain find_spots page
    has_find_spots = all([nb.has_page('find_spots') for nb in nb_list])
    if not has_find_spots:
        return master_nb
    nbp_find_spots_list = [nb.find_spots for nb in nb_list]
    master_nbp_find_spots = merge_find_spots(nbp_find_spots_list, master_nbp_basic)
    master_nb += master_nbp_find_spots
    master_nb.find_spots.finalized = True

    # Check the notebooks contain register page
    has_register = all([nb.has_page('register') for nb in nb_list])
    if not has_register:
        return master_nb
    nbp_register_list = [nb.register for nb in nb_list]
    master_nbp_register = merge_register(nbp_register_list, master_nbp_basic)
    master_nb += master_nbp_register
    master_nb.register.finalized = True

    # Check the notebooks contain register_debug page
    has_register_debug = all([nb.has_page('register_debug') for nb in nb_list])
    if not has_register_debug:
        return master_nb
    nbp_register_debug_list = [nb.register_debug for nb in nb_list]
    master_nbp_register_debug = merge_register_debug(nbp_register_debug_list, master_nbp_basic)
    master_nb += master_nbp_register_debug

    return master_nb

    # Check the notebooks contain register page


def merge_extract(nbp_extract_list: List[NotebookPage], master_nbp_basic: NotebookPage) -> NotebookPage:
    """
    Merge a list of single tile nbp_extract into one multitile nbp_extract.
    
    Args:
        nbp_extract_list (list of NotebookPage): list of extract pages to be combined.
        master_nbp_basic (NotebookPage): nbp_basic page for master notebook.

    Returns:
        master_nbp_extract: multi-tile nbp_extract page.
    
    Raises:
        AssertionError: merge of the given extract list is not possible based on their values.
    """
    for i in range(len(nbp_extract_list) - 1):
        assert nbp_extract_list[i].file_type == nbp_extract_list[i + 1].file_type, \
            "file_type must be the same for every extract page to merge them"
        assert nbp_extract_list[i].continuous_dapi == nbp_extract_list[i + 1].continuous_dapi
        assert nbp_extract_list[i].software_version == nbp_extract_list[i + 1].software_version, \
            "software_version must be the same to merge extract pages"
        assert nbp_extract_list[i].revision_hash == nbp_extract_list[i + 1].revision_hash, \
            "revision_hash must be the same to merge extract pages"

    # Create a master notebook extract page
    master_nbp_extract = NotebookPage('extract')

    # Add the extraction file type. Should be same across notebooks.
    master_nbp_extract.file_type = nbp_extract_list[0].file_type
    master_nbp_extract.continuous_dapi = nbp_extract_list[0].continuous_dapi
    master_nbp_extract.software_version = nbp_extract_list[0].software_version
    master_nbp_extract.revision_hash = nbp_extract_list[0].revision_hash

    return master_nbp_extract


def merge_extract_debug(nbp_extract_debug_list: List[NotebookPage], master_nbp_basic: NotebookPage) -> NotebookPage:
    """
    Merge a list of single tile 'extract_debug' notebook pages into one multi-tile 'extract_debug' notebook page.
    
    Args:
        nbp_extract_debug_list: List of extract_debug pages to be combined
        master_nbp_basic: nbp_basic page for master notebook

    Returns:
        master_nbp_extract: multi-tile nbp_extract_debug page
        
    Raises:
        AssertionError: merging the given list of NotebookPages is not possible.
    """
    # Create a master notebook extract page
    master_nbp_extract_debug = NotebookPage('extract_debug')

    time_taken = 0
    pixel_unique_values = np.full(
        (
            master_nbp_basic.n_tiles,
            master_nbp_basic.n_rounds + master_nbp_basic.n_extra_rounds,
            master_nbp_basic.n_channels,
            np.iinfo(np.uint16).max,
        ),
        fill_value=0,
        dtype=int, 
    )
    pixel_unique_counts = pixel_unique_values.copy()
    for i, tile in enumerate(master_nbp_basic.use_tiles):
        time_taken += nbp_extract_debug_list[i].time_taken
        pixel_unique_values[tile] = nbp_extract_debug_list[i].pixel_unique_values[tile]
        pixel_unique_counts[tile] = nbp_extract_debug_list[i].pixel_unique_counts[tile]
    master_nbp_extract_debug.time_taken = time_taken
    master_nbp_extract_debug.pixel_unique_values = pixel_unique_values
    master_nbp_extract_debug.pixel_unique_counts = pixel_unique_counts

    return master_nbp_extract_debug


def merge_filter(nbp_filter_list: List[NotebookPage], master_nbp_basic: NotebookPage) -> NotebookPage:
    """
    Merge 'filter' notebook pages together for each tile based on the tile indices indicated in the master 'basic_info' 
    notebook page.

    Args:
        nbp_filter_list (list[NotebookPage]): list of filter notebook pages to merge.
        master_nbp_basic (NotebookPage): master 'basic_info' notebook page.

    Returns:
        NotebookPage: merged 'filter' notebook page.

    Raises:
        AssertionError: failed to merge filter notebook pages based on variable mismatch.
    """
    assert len(nbp_filter_list) > 1, "Need at least two 'filter' notebook pages to merge"
    assert len(master_nbp_basic.use_tiles) > 1, "Need at least two tiles in master_nbp_basic.use_tiles"
    assert master_nbp_basic.n_tiles > 1, "Need at least two tiles in master_nbp_basic.n_tiles"

    master_nbp_filter = NotebookPage("filter")

    for i in range(len(nbp_filter_list) - 1):
        assert np.allclose(nbp_filter_list[i].hist_values, nbp_filter_list[i + 1].hist_values), \
            "hist_values must be the same for every filter notebook page to merge them"
        assert nbp_filter_list[i].software_version == nbp_filter_list[i + 1].software_version, \
            "software_version must be the same to merge filter pages"
        assert nbp_filter_list[i].revision_hash == nbp_filter_list[i + 1].revision_hash, \
            "revision_hash must be the same to merge filter pages"
    
    master_nbp_filter.hist_values = nbp_filter_list[0].hist_values

    use_tiles = master_nbp_basic.use_tiles
    n_tiles, n_rounds, n_channels = master_nbp_basic.n_tiles, master_nbp_basic.n_rounds, master_nbp_basic.n_channels
    
    # Initialise auto_thresh and hist_counts
    auto_thresh = np.zeros(
        (n_tiles, n_rounds + master_nbp_basic.n_extra_rounds, n_channels), 
        dtype=int, 
    )
    hist_counts = np.zeros_like(nbp_filter_list[0].hist_counts, dtype=int)
    bg_scale = np.zeros((n_tiles, n_rounds, n_channels), dtype=float)
    # Add the tiles we are using.
    # For hist_counts we need to add all these pages as this is supposed to be across all tiles
    for i, tile in enumerate(use_tiles):
        auto_thresh[tile] = nbp_filter_list[i].auto_thresh[tile]
        hist_counts += nbp_filter_list[i].hist_counts
        if bg_scale is not None and nbp_filter_list[i].bg_scale is not None:
            bg_scale[tile] = nbp_filter_list[i].bg_scale[tile]
        else:
            bg_scale = None

    # Add to the master notebook page
    master_nbp_filter.auto_thresh = auto_thresh
    master_nbp_filter.hist_counts = hist_counts
    master_nbp_filter.bg_scale = bg_scale
    master_nbp_filter.software_version = nbp_filter_list[0].software_version
    master_nbp_filter.revision_hash = nbp_filter_list[0].revision_hash
    
    return master_nbp_filter


def merge_filter_debug(nbp_filter_debug_list: List[NotebookPage], master_nbp_basic: NotebookPage) -> NotebookPage:
    """
    Merge 'filter_debug' notebook pages together for each tile based on the tile indices indicated in the master 
    'basic_info' notebook page.

    Args:
        nbp_filter_debug_list (list[NotebookPage]): list of filter notebook pages to merge.
        master_nbp_basic (NotebookPage): master 'basic_info' notebook page.

    Returns:
        NotebookPage: merged 'filter_debug' notebook page.

    Raises:
        AssertionError: failed to merge filter notebook pages based on variable mismatch.
    """
    for i in range(len(nbp_filter_debug_list) - 1):
        if nbp_filter_debug_list[i].r_dapi is not None:
            assert np.isclose(nbp_filter_debug_list[i].r_dapi, nbp_filter_debug_list[i + 1].r_dapi)
        assert np.isclose(nbp_filter_debug_list[i].z_info, nbp_filter_debug_list[i + 1].z_info)
        if nbp_filter_debug_list[i].psf is not None:
            warnings.warn(
                "Additional data about the psf is not saved correctly in filter_debug, this is only used for " \
                + "diagnostics"
            )
            #TODO: Check psf's for equality if deconvolve is set to true

    master_nbp_filter_debug = NotebookPage("filter_debug")
    
    master_nbp_filter_debug.r_dapi = nbp_filter_debug_list[0].r_dapi
    master_nbp_filter_debug.z_info = nbp_filter_debug_list[0].z_info
    
    # Extract tiles that we're using
    use_tiles = master_nbp_basic.use_tiles
    n_tiles, n_rounds, n_channels = master_nbp_basic.n_tiles, master_nbp_basic.n_rounds, master_nbp_basic.n_channels

    # Initialise non-trivial variables
    n_clip_pixels = np.zeros((n_tiles, n_rounds + master_nbp_basic.n_extra_rounds, n_channels), dtype=int)
    clip_extract_scale = np.zeros((n_tiles, n_rounds + master_nbp_basic.n_extra_rounds, n_channels))
    time_taken = 0.
    # Loop over all tiles in use and assign these their proper values
    for i, tile in enumerate(use_tiles):
        n_clip_pixels[tile] = nbp_filter_debug_list[i].n_clip_pixels[tile]
        clip_extract_scale[tile] = nbp_filter_debug_list[i].clip_extract_scale[tile]
        time_taken += nbp_filter_debug_list[i].time_taken

    # Assign these values to the notebook page
    master_nbp_filter_debug.n_clip_pixels = n_clip_pixels
    master_nbp_filter_debug.clip_extract_scale = clip_extract_scale
    master_nbp_filter_debug.time_taken = time_taken
    
    return master_nbp_filter_debug


def merge_find_spots(nbp_find_spots_list: List[NotebookPage], master_nbp_basic: NotebookPage) -> NotebookPage:
    """
    Merge a list of single tile nbp_find_spots into one multi-tile nbp_find_spots.
    
    Args:
        nbp_find_spots_list: List of find_spots pages to be combined.
        master_nbp_basic: nbp_basic page for master notebook. Must contain information `use_tiles`, `n_tiles`, 
            `n_rounds`, `n_channels` based on the list of 'find_spots' pages.

    Returns:
        master_nbp_find_spots: multi-tile 'find_spots' page.
        
    Raises:
        AssertionError: merging the given list of NotebookPages is not possible.
    """
    # Create a master notebook find_spots page to append all tile pages to
    master_nbp_find_spots = NotebookPage('find_spots')

    # Extract tiles that we're using
    use_tiles = master_nbp_basic.use_tiles
    n_tiles, n_rounds, n_channels = master_nbp_basic.n_tiles, master_nbp_basic.n_rounds, master_nbp_basic.n_channels

    for i in range(len(nbp_find_spots_list) - 1):
        assert nbp_find_spots_list[i].software_version == nbp_find_spots_list[i + 1].software_version, \
            "software_version must be the same to merge find_spots pages"
        assert nbp_find_spots_list[i].revision_hash == nbp_find_spots_list[i + 1].revision_hash, \
            "revision_hash must be the same to merge find_spots pages"

    # Now populate all the parameters
    spot_yxz = np.zeros((0, 3), dtype=int)
    isolated_spots = np.zeros(0, dtype=bool)
    spot_no = np.zeros((n_tiles, n_rounds + master_nbp_basic.n_extra_rounds, n_channels), dtype=np.int32)
    isolation_thresh = np.zeros(n_tiles)
    for i, tile in enumerate(use_tiles):
        spot_yxz = np.vstack((spot_yxz, nbp_find_spots_list[i].spot_yxz))
        spot_no[tile] = nbp_find_spots_list[i].spot_no[tile]
        isolated_spots = np.append(isolated_spots, nbp_find_spots_list[i].isolated_spots)
        isolation_thresh[tile] = nbp_find_spots_list[i].isolation_thresh[tile]

    # Add these all to the notebook page
    master_nbp_find_spots.spot_yxz = spot_yxz
    master_nbp_find_spots.spot_no = spot_no
    master_nbp_find_spots.isolated_spots = isolated_spots
    master_nbp_find_spots.isolation_thresh = isolation_thresh
    master_nbp_find_spots.software_version = nbp_find_spots_list[0].software_version
    master_nbp_find_spots.revision_hash = nbp_find_spots_list[0].revision_hash

    return master_nbp_find_spots


def merge_register(nbp_register_list, master_nbp_basic) -> NotebookPage:
    """
    Merge a list of single tile nbp_register into one multi-tile nbp_register.
    
    Args:
        nbp_register_list: list of register pages to be combined.
        master_nbp_basic: 'basic_info' page for master notebook to base the combined NotebookPage on.

    Returns:
        master_nbp_extract: multi-tile nbp_register page.
        
    Raises:
        AssertionError: merging the given list of NotebookPages is not possible.
    """
    # Create a master notebook reg page
    master_nbp_register = NotebookPage('register')

    # Extract tiles that we're using
    use_tiles = master_nbp_basic.use_tiles
    n_tiles, n_rounds, n_channels = master_nbp_basic.n_tiles, master_nbp_basic.n_rounds, master_nbp_basic.n_channels

    for i in range(len(nbp_register_list) - 1):
        assert nbp_register_list[i].software_version == nbp_register_list[i + 1].software_version, \
            "software_version must be the same to merge register pages"
        assert nbp_register_list[i].revision_hash == nbp_register_list[i + 1].revision_hash, \
            "revision_hash must be the same to merge register pages"

    # initialise all our variables
    initial_transform = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    transform = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    round_transform = np.zeros((n_tiles, n_rounds, 4, 3))
    channel_transform = np.zeros((n_tiles, n_channels, 4, 3))

    # Loop over all tiles in use and populate these arrays
    for i, tile in enumerate(use_tiles):
        initial_transform[tile] = nbp_register_list[i].initial_transform[tile]
        transform[tile] = nbp_register_list[i].transform[tile]
        round_transform[tile] = nbp_register_list[i].round_transform[tile]
        channel_transform[tile] = nbp_register_list[i].channel_transform[tile]

    # Now assign these to the master notebook
    master_nbp_register.start_transform = initial_transform
    master_nbp_register.transform = transform
    master_nbp_register.round_transform = round_transform
    master_nbp_register.channel_transform = channel_transform
    master_nbp_register.software_version = nbp_register_list[0].software_version
    master_nbp_register.revision_hash = nbp_register_list[0].revision_hash

    return master_nbp_register


def merge_register_debug(
    nbp_register_debug_list: List[NotebookPage], master_nbp_basic: NotebookPage, 
    ) -> NotebookPage:
    """
    Merge a list of single-tile 'register_debug' NotebookPages into one multi-tile NotebookPage.

    Args:
        nbp_register_debug_list (list of NotebookPage): list of finalised 'register_debug' NotebookPages to combine.
        master_nbp_basic: 'basic_info' page for master notebook to base the combined NotebookPage on.

    Returns:
        NotebookPage: multi-tile 'register_debug' NotebookPage.
    
    Raises:
        AssertionError: merging the given list of NotebookPages is not possible.
    """
    # Create a master notebook reg page
    master_nbp_register_debug = NotebookPage('register_debug')
    
    use_tiles = master_nbp_basic.use_tiles
    n_tiles, n_rounds, n_channels = master_nbp_basic.n_tiles, master_nbp_basic.n_rounds, master_nbp_basic.n_channels
    
    for i in range(len(nbp_register_debug_list) - 1):
        n_iters = nbp_register_debug_list[i].n_matches.shape[3]
        zyx_subvols = nbp_register_debug_list[i].round_shift.shape[2]
        channel_transform = nbp_register_debug_list[i].channel_transform
        assert zyx_subvols == nbp_register_debug_list[i + 1].round_shift.shape[2], \
            "subvol sizes are not the same for every 'register_debug' NotebookPage"
        assert n_iters == nbp_register_debug_list[i + 1].n_matches.shape[3], \
            "n_iters is not the same value for every 'register_debug' NotebookPage"
        assert n_iters == nbp_register_debug_list[i + 1].mse.shape[3], \
            "n_iters is not the same value for every 'register_debug' NotebookPage"
        assert np.allclose(channel_transform, nbp_register_debug_list[i + 1].channel_transform), \
            "channel_transform is not the same for every 'register_debug' NotebookPage"
    
    n_matches = np.zeros((n_tiles, n_rounds, n_channels, n_iters))
    mse = np.zeros((n_tiles, n_rounds, n_channels, n_iters))
    converged = np.zeros((n_tiles, n_rounds, n_channels))
    round_shift = np.zeros((n_tiles, n_rounds, zyx_subvols, 3))
    round_shift_corr = np.zeros((n_tiles, n_rounds, zyx_subvols))
    position = np.zeros((n_tiles, n_rounds, zyx_subvols, 3))
    round_transfrom_raw = np.zeros((n_tiles, n_rounds, 3, 4))

    for i, tile in enumerate(use_tiles):
        n_matches[tile] = nbp_register_debug_list[i].n_matches[tile]
        mse[tile] = nbp_register_debug_list[i].mse[tile]
        converged[tile] = nbp_register_debug_list[i].converged[tile]
        round_shift[tile] = nbp_register_debug_list[i].round_shift[tile]
        round_shift_corr[tile] = nbp_register_debug_list[i].round_shift_corr[tile]
        position[tile] = nbp_register_debug_list[i].position[tile]
        round_transfrom_raw[tile] = nbp_register_debug_list[i].round_transfrom_raw[tile]
    
    master_nbp_register_debug.channel_transform = channel_transform
    master_nbp_register_debug.n_matches = n_matches
    master_nbp_register_debug.n_matches = mse
    master_nbp_register_debug.n_matches = converged
    master_nbp_register_debug.n_matches = round_shift
    master_nbp_register_debug.n_matches = round_shift_corr
    master_nbp_register_debug.n_matches = position
    master_nbp_register_debug.n_matches = round_transfrom_raw
    
    return master_nbp_register_debug


def split_by_tiles(master_notebook: Notebook) -> List[Notebook]:
    """
    Split a notebook into separate notebooks for each tile. This can only be done with a notebook that contains 
    'basic_info', 'file_names', and 'scale' pages. The split notebook for tile `t` is saved with name `'notebook_t{t}'`.

    Args:
        master_notebook (Notebook): notebook to split.

    Returns:
        List[Notebook]: deep copy of notebook for each tile in `basic_info.use_tiles`. Notebook at index `i` 
            corresponds to `basic_info.use_tiles[i]`.
    """
    assert master_notebook.has_page('basic_info'), "Require 'basic_info' notebook page to split by tiles"
    assert master_notebook.has_page('file_names'), "Require 'file_names' notebook page to split by tiles"
    assert master_notebook.has_page('scale'), "Require 'scale' notebook page to split by tiles"
    assert not master_notebook.has_page('extract'), "Notebook cannot contain 'extract' notebook page"
    
    output = []
    notebook_dir = PurePath(master_notebook._file).parent
    use_tiles = master_notebook.basic_info.use_tiles
    for tile in use_tiles:
        new_notebook = copy.deepcopy(master_notebook)
        new_notebook._created_time = time.time()
        new_notebook._file = os.path.join(notebook_dir, f"notebook_t{tile}")
        
        new_notebook.basic_info.finalized = False
        del new_notebook.basic_info.use_tiles, new_notebook.basic_info.n_tiles
        new_notebook.basic_info.use_tiles = [tile]
        new_notebook.basic_info.n_tiles = tile + 1
        new_notebook.basic_info.finalized = True

        new_notebook.file_names.finalized = False
        old_spot_details_path = PurePath(new_notebook.file_names.spot_details_info)
        del new_notebook.file_names.spot_details_info
        new_notebook.file_names.spot_details_info = str(
            PurePath(
                old_spot_details_path.parent, 
                f"spot_details_info_t{tile}.npz", 
            )
        )
        new_notebook.file_names.finalized = True

        output.append(new_notebook)
    return output


def split_stitch(master_nbp_stitch, nb_list):
    """
    Function to split the stitch page from the master notebook to the individual notebooks. Does not return anything
    but simply adds the pages to the single tile notebooks.

    Args:
        master_nbp_stitch: stitch page of master notebook
        nb_list: list of single tile notebooks

    Returns:
        N/A
    """
    raise NotImplementedError("This is outdated, why do you need this function?")
    # Cannot just take a page from a notebook and add it to another unfortunately, have to manually copy arguments
    for i in range(len(nb_list)):

        # initialise notebook page for each individual stitch page
        nbp = NotebookPage('stitch')
        # First 4 keys are 'finalized', '_times', 'name', '_time_created', we do not need to set these.
        key_list = list(master_nbp_stitch.__dict__)[4:]
        # For remaining arguments set to None but then update tile_origin
        for key in key_list:
            nbp.__setattr__(key=key, value=None)
        origin = np.zeros((1, 3))
        origin[0] = master_nbp_stitch.tile_origin[i]
        # Add this updated tile origin
        nbp.tile_origin = origin
        # Add notebook page
        nb_list[i] += nbp


def split_call_spots(master_nbp_call_spots, nb_list):
    """
    Function to split the call_spots page from the master notebook to the individual notebooks. Does not return anything
    but simply adds the pages to the single tile notebooks.

    Args:
        master_nbp_call_spots: call_spots page of master notebook
        nb_list: list of single tile notebooks

    Returns:
        N/A
    """
    raise NotImplementedError("This is outdated, why do you need this function?")
    # I was lying earlier when I said we couldn't copy a page verbatim. Just have to use a trick
    for i in range(len(nb_list)):
        # This allows us to add the page to another notebook
        master_nbp_call_spots.finalized = False
        # Adding the page to a notebook sets finalize = true so have to keep this condition change in the loop
        nb_list[i] += master_nbp_call_spots


def split_ref_spots(master_nbp_ref_spots, nb_list):
    """
    Function to split the ref_spots page from the master notebook to the individual notebooks. Does not return anything
    but simply adds the pages to the single tile notebooks.

    Args:
        master_nbp_ref_spots: refl_spots page of master notebook
        nb_list: list of single tile notebooks

    Returns:
        N/A
    """
    raise NotImplementedError("This is outdated, why do you need this function?")
    # Load spot_indices where one tile ends and another begins
    num_spots = [nb.find_spots.spot_details.shape[0] for nb in nb_list]
    num_notebooks = len(nb_list)

    # Sum all these to get indices
    index = [sum(num_spots[:i]) for i in range(num_notebooks + 1)]

    # First 4 keys are 'finalized', '_times', 'name', '_time_created', we do not need to set these.
    key_list = list(master_nbp_ref_spots.__dict__)[4:]

    # Loop through all notebooks
    for i in range(num_notebooks):
        # Loop through all parameters to be set
        for key in key_list:
            # initialise nbp
            nbp = NotebookPage('ref_spots')
            # Set attribute to be all spots in the ith tile
            nbp.__setattr__(key=key, value=master_nbp_ref_spots.__getattribute__(key)[index[i]:index[i+1]])
        nb_list[i] += nbp
