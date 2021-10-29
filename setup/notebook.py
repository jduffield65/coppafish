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
    ("string",
     lambda x : isinstance(x, (str, np.str_)),
     lambda x : str(x[()]),
    ),
    ("ndarray",
     lambda x : isinstance(x, np.ndarray),
     lambda x : x,
    ),
    ("int",
     lambda x : isinstance(x, (int, np.int_)),
     lambda x : int(x[()]),
    ),
    ("number",
     lambda x : np.isreal(x) is True, # is True guards against isreal returning an array
     lambda x : float(x[()]),
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
    for n,_,f in TYPES:
        if n == typ:
            return f(val)
    raise TypeError(f"Key {key!r} has type {typ!r}, "\
                    "but we don't know how to decode that.  "\
                    "Please use one of the following: {[t[0] for t in TYPES]}")

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
    for n,f,_ in TYPES:
        if f(val):
            return n
    raise TypeError(f"Key {key!r} has value {val!r} which "\
                    "is of type {type(val)}, which is invalid.  "\
                    "Please use one of the following: {[t[0] for t in TYPES]}")


class Notebook:
    """A write-only file-synchronized class to keep track of ISS results.

    The Notebook object stores all of the outputs of the script.  Almost all
    information saved in the Notebook is encapsulated within "pages", from the
    NotebookPage object.  To add a NotebookPage object to a notebook, use the
    "add_page" method.  You can add pages, but you cannot remove them.  Pages
    can be referenced by their name using the square bracket (subscript)
    notation.  In addition to saving pages, it also saves the contents of the
    config file, and the time at which the notebook and each page was created.

    To create a Notebook, pass it the path to the file where the Notebook is to
    be stored (`notebook_file`), and the path to the configuration file
    (`config_file`).  If `notebook_file` already exists, the notebook located
    at this path will be loaded.  If not, a new file will be created as soon as
    the first data is written to the Notebook.

    Example:

        nb = Notebook("nbfile.npz")
        nbp = NotebookPage("pagename")
        nbp["var"] = 1
        nb.add_page(nbp)
        assert nb["pagename"]["var"] == 1

    Because it is automatically saved to the disk, you can close Python, reopen
    it, and do the following:

        nb2 = Notebook("nbfile.npz")
        assert nb2["pagename"]["var"] == 1
    """
    _SEP = "_-_" # Separator between notbook page name and item name when saving to file
    _ADDEDMETA = "TIME_CREATED" # Key for notebook created time
    _CONFIGMETA = "CONFIGFILE" # Key for notebook created time
    _NBMETA = "NOTEBOOKMETA" # Key for metadata about the entire notebook
    def __init__(self, notebook_file, config_file):
        # numpy isn't compatible with npz files which do not end in the suffiz
        # .npz.  If one isn't there, it will add the extension automatically.
        # We do the same thing here.
        if not notebook_file.endswith(".npz"):
            notebook_file = notebook_file + ".npz"
        # Note that the ordering of _pages may change across saves and loads,
        # but the order will always correspond to the order of _pages_times
        self._file = notebook_file
        # Read the config file, but don't assign anything yet.  Here, we just
        # save a copy of the config file.  This isn't the main place the config
        # file should be read from.
        with open(config_file, 'r') as f:
            read_config = f.read()
        # If the file already exists, initialize the Notebook object from this
        # file.  Otherwise, initialize it empty.
        if os.path.isfile(self._file):
            self._pages, self._pages_times, self._created_time, self._config = self.from_file(self._file)
            if read_config != self._config:
                raise SystemError("Passed config file is not the same as the saved config file")
        else:
            self._pages = []
            self._pages_times = []
            self._created_time = time.time()
            self._config = read_config
    def __eq__(self, other):
        """Test if two Notebooks are identical

        For two Notebooks to be identical, all aspects must be the same,
        excluding the ordering of the pages, and the filename.  All timestamps
        must also be identical.
        """
        if self._created_time != other._created_time:
            return False
        if self._config != other._config:
            return False
        if len(self._pages) != len(other._pages):
            return False
        # There is no cleaner way to write argsort in pure python, and we don't
        # want to use numpy for this
        argsort = lambda x : list(zip(*sorted(zip(x, range(0, 10000)))))[1]
        # We need to make sure the way self._pages (and, by extension,
        # self._pages_times) is sorted does not matter.
        order_self = argsort(p.name for p in self._pages)
        order_other = argsort(p.name for p in other._pages)
        for i_self, i_other in zip(order_self, order_other):
            if self._pages[i_self] != other._pages[i_other]:
                return False
            if self._pages_times[i_self] != other._pages_times[i_other]:
                return False
        return True
    def __len__(self):
        """Return the number of pages in the Notebook"""
        return len(self._pages)
    def add_page(self, page):
        """Insert the page `page` into the Notebook.

        This function automatically triggers a save.
        """
        if self._SEP in page.name:
            raise NameError("The separator {self._SEP} may not be in the page's name")
        if page.finalized:
            raise ValueError("Page already added to a Notebook, cannot add twice")
        if any(page.name == p.name for p in self._pages):
            raise ValueError("Cannot add two pages with the same name")
        page.finalized = True
        self._pages.append(page)
        self._pages_times.append(time.time())
        self.save()
    def __iadd__(self, other):
        """Syntactic sugar for the add_page method"""
        if not isinstance(other, NotebookPage):
            raise ValueError("Only NotebookPage objects may be added to a notebook.")
        self.add_page(other)
        return self
    def __getitem__(self, name):
        """Access the contents of the page using the square bracket notation

        For a Notebook nb, this implements the notation nb[pagename] where,
        pagename is the name of a page in the Notebook.  Note that pagename is
        specified when creating the NotebookPage object and is never explicitly
        specified when adding the page to the Notebook.  In combination with
        the square bracket notation of the NotebookPage, this allows notation
        like nb[pagename][keyname].
        """
        for p in self._pages:
            if p.name == name:
                return p
        raise KeyError(f"No Notebook page with the name {name!r}.")
    def version_hash(self):
        """A short string representing the file version.

        Since there are many possible page names and entry names within those
        pages, that means there are many, many possible file versions based on
        different versions of the code.  Rather than try to keep track of these
        versions and appropriately increment some centralized counter, we
        generate a short string which is a hash of the page names and the names
        of the entries in that page.  This way, it is possible to see if two
        logs were generated using the same version of the software.  (Of
        course, it assumes that no fields are ever set conditionally.)
        """
        s = ""
        for p in self._pages:
            s += p.name + "\n\n"
            s += "\n".join(sorted(p._results.keys()))
        return hashlib.md5(bytes(s, "utf8")).hexdigest()
    def save(self):
        """Save the Notebook to a file"""
        d = {}
        # Diagnostic information about how long the save took.  We can probably
        # take this out, or else set it at a higher debug level via warnings
        # module.
        save_start_time = time.time()
        for i,p in enumerate(self._pages):
            pd = p.to_serial_dict()
            name = p.name
            for k,v in pd.items():
                d[name+self._SEP+k] = v
            d[name+self._SEP+self._ADDEDMETA] = self._pages_times[i]
        d[self._NBMETA+self._SEP+self._ADDEDMETA] = self._created_time
        d[self._NBMETA+self._SEP+self._CONFIGMETA] = self._config
        np.savez_compressed(self._file, **d)
        # Finishing the diagnostics described above
        print(f"Notebook saved: took {time.time()-save_start_time} seconds")

    @classmethod
    def from_file(cls, fn):
        """Read a Notebook from a file

        The only argument is `fn`, the filename of the saved Notebook to load.

        This returns a tuple of three objects:

        - A list of NotebookPage objects
        - A list of timestamps, of identical length to the list of NotebookPage objects
        - A timestamp for the time the Notebook was created.
        - A string of the config file
        """
        # Right now we won't use lazy loading.  One problem with lazy loading
        # is that we must keep the file handle open.  We would rather not do
        # this, because if we write to the file, it will get screwed up, and if
        # there is a network issue, it will also mess things up.  I can't
        # imagine that loading the notebook will be a perferomance bottleneck,
        # but if it is, we can rethink this decision.  It should be pretty easy
        # to lazy load the pages, but eager load everything in the page.
        f = np.load(fn)
        keys = list(f.keys())
        page_items = {}
        page_times = {}
        created_time = None
        for pk in keys:
            p,k = pk.split(cls._SEP, 1)
            if p == cls._NBMETA:
                if k == cls._ADDEDMETA:
                    created_time = f[pk]
                    continue
                if k == cls._CONFIGMETA:
                    config_file = f[pk]
                    continue
            if k == cls._ADDEDMETA:
                page_times[p] = f[pk]
                continue
            if p not in page_items.keys():
                page_items[p] = {}
            page_items[p][k] = f[pk]
        pages = [NotebookPage.from_serial_dict(page_items[d]) for d in sorted(page_items.keys())]
        pages_times = [page_times[d] for d in sorted(page_items.keys())]
        assert len(pages) == len(page_times), "Invalid file, lengths don't match"
        assert created_time is not None, "Invalid file, invalid created date"
        return (pages, pages_times, created_time, config_file)



class NotebookPage:
    """A page, to be added to a Notebook object

    Expected usage is for a NotebookPage to be created at the beginning of a
    large step in the analysis pipeline.  The name of the page should reflect
    its function, and it will be used as the indexing key when it is added to a
    Notebook.  The NotebookPage should be created at the beginning of the step
    in the pipeline, because then the timestamp will be more meaningful.  As
    results are computed, they should be added.  This will provide a timestamp
    for each of the results as well.  Then, at the end, the pipeline step should return
    a NotebookPage, which can then be added to the Notebook.

    Example:

        nbp = NotebookPage("extract_and_filter")
        nbp["scale_factor"] = 10
        ...
        return nbp
    """
    _PAGEMETA = "PAGEINFO" # Filename for metadata about the page
    _TIMEMETA = "___TIME" # Filename suffix for timestamp information
    _TYPEMETA = "___TYPE" # Filename suffix for type information
    def __init__(self, name):
        self.name = name
        self._time_created = time.time()
        self._results = {}
        self._times = {}
        self.finalized = False # Set to true when added to a Notebook
    def __eq__(self, other):
        """Test for equality using the == syntax.

        To be honest, I don't know why you would ever need this, but it is very
        useful for writing unit tests, so here it is.
        """
        if not isinstance(other, self.__class__):
            return False
        if self.name != other.name:
            return False
        if self._time_created != other._time_created:
            return False
        for k,v in self._results.items():
            if k not in other._results or not np.array_equal(v, other._results[k]):
                return False
        for k,v in other._results.items():
            if k not in self._results or not np.array_equal(v, self._results[k]):
                return False
        for k,v in self._times.items():
            if k not in other._times or v != other._times[k]:
                return False
        return True
    def __len__(self):
        """Return the number of results in the NotebookPage"""
        return len(self._results)
    def __setitem__(self, key, value):
        """Add an item to the notebook page.

        For a NotebookPage object nbp, this handles the syntax nbp[key] = value.
        It checks the key and value for validity, and then adds them to the
        notebook.  Specifically, it implements a write-once mechanism.
        """
        if self.finalized:
            raise ValueError("This NotebookPage has already been added to a Notebook, no more values can be added.")
        assert isinstance(key, str), f"NotebookPage key {key!r} must be a string, not {type(key)}"
        _get_type(key, value)
        if key in self._results.keys():
            raise ValueError(f"Cannot assign {key} = {value!r} to the notebook page, key already exists")
        self._results[key] = value
        self._times[key] = time.time()
    def __getitem__(self, key):
        """Return an item from the NotebookPage.

        For a NotebookPage object, lp, this handles the syntax lp[key] for
        reading, not writing, values.
        """
        if key not in self._results.keys():
            raise ValueError(f"Cannot access {key!r} in the notebook, key doesn't exist")
        return self._results[key]
    def to_serial_dict(self):
        """Convert to a dictionary which can be written to a file.

        In general, this function shouldn't need to be called other than within
        a Notebook object.
        """
        keys = {}
        keys[self._PAGEMETA] = self.name
        keys[self._PAGEMETA+self._TIMEMETA] = self._time_created
        for rn,r in self._results.items():
            keys[rn] = r
            keys[rn+self._TIMEMETA] = self._times[rn]
            keys[rn+self._TYPEMETA] = _get_type(rn, r)
        return keys
    @classmethod
    def from_serial_dict(cls, d):
        """Convert from a dictionary to a NotebookPage object

        In general, this function shouldn't need to be called other than within
        a Notebook object.
        """
        # Note that this method will need to be updated if you update the
        # constructor.
        name = str(d[cls._PAGEMETA][()])
        n = cls(name)
        n._time_created = d[cls._PAGEMETA+cls._TIMEMETA]
        for k in d.keys():
            # If we've already dealt with the key, skip it.
            if k.startswith(cls._PAGEMETA): continue
            # Each key has an associated "time" and "type" key.  We deal with
            # the time and type keys separately when dealing with the main key.
            if k.endswith(cls._TIMEMETA): continue
            if k.endswith(cls._TYPEMETA): continue
            # Now that we have a real key, add it to the page.
            n._results[k] = _decode_type(k, d[k], str(d[k+cls._TYPEMETA][()]))
            n._times[k] = d[k+cls._TIMEMETA]
        return n


