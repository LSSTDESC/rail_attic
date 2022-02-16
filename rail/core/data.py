"""Rail-specific data management"""

import os
import tables_io
import qp

class DataHandle:
    """Class to act as a handle for a bit of data.  Associated it with a file and
    providing tools to read & write it to that file

    Parameters
    ----------
    tag : str
        The tag under which this data handle can be found in the store
    data : any or None
        The associated data
    path : str or None
        The path to the associated file
    creator : str or None
        The name of the stage that created this data handle
    """
    suffix = ''

    def __init__(self, tag, data=None, path=None, creator=None):
        """Constructor """
        self.tag = tag
        self.data = data
        self.path = path
        self.creator = creator

    def open(self, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        if self.path is None:
            raise ValueError("TableHandle.open() called but path has not been specified")
        return self._open(self.path, **kwargs)

    @classmethod
    def _open(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._open")

    def read(self, force=False, **kwargs):
        """Read and return the data from the associated file """
        if self.data is not None and not force:
            return self.data
        self.data = self._read(self.path, **kwargs)
        return self.data

    @classmethod
    def _read(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._read")

    def write(self, **kwargs):
        """Write the data to the associatied file """
        if self.path is None:
            raise ValueError("TableHandle.write() called but path has not been specified")
        if self.data is None:
            raise ValueError(f"TableHandle.write() called for path {self.path} with no data")
        return self._write(self.data, self.path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        raise NotImplementedError("DataHandle._write")

    def iterator(self, **kwargs):
        """Iterator over the data"""
        #if self.data is not None:
        #    for i in range(1):
        #        yield i, -1, self.data
        return self._iterator(self.path, **kwargs)

    @classmethod
    def _iterator(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._iterator")

    @property
    def has_data(self):
        """Return true if the data for this handle are loaded """
        return self.data is not None

    @property
    def has_path(self):
        """Return true if the path for the associated file is defined """
        return self.path is not None

    @property
    def is_written(self):
        """Return true if the associated file has been written """
        if self.path is None:
            return False
        return os.path.exists(self.path)

    def __str__(self):
        s = f"{type(self)} "
        if self.has_path:
            s += f"{self.path}, ("
        else:
            s += "None, ("
        if self.is_written:
            s += "w"
        if self.has_data:
            s += "d"
        s += ")"
        return s

    @classmethod
    def make_name(cls, tag):
        """Construct and return file name for a particular data tag """
        if cls.suffix:
            return f"{tag}.{cls.suffix}"
        else:
            return tag


class TableHandle(DataHandle):
    """DataHandle for single tables of data
    """
    suffix = None

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.open(path, **kwargs)  #pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return tables_io.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associatied file """
        return tables_io.write(data, path, **kwargs)

    @classmethod
    def _iterator(cls, path, **kwargs):
        """Iterate over the data"""
        return tables_io.iteratorNative(path, **kwargs)


class QPHandle(DataHandle):
    """DataHandle for qp ensembles
    """
    suffix = 'fits'

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.open(path)  #pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return qp.read(path)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associatied file """
        return data.write_to(path)


class DataFile:
    """
    A class representing a DataFile to be made by pipeline stages
    and passed on to subsequent ones.

    DataFile itself should not be instantiated - instead subclasses
    should be defined for different file types.

    These subclasses are used in the definition of pipeline stages
    to indicate what kind of file is expected.  The "suffix" attribute,
    which must be defined on subclasses, indicates the file suffix.

    The open method, which can optionally be overridden, is used by the
    machinery of the PipelineStage class to open an input our output
    named by a tag.

    """
    suffix = ''

    def __init__(self, path, mode, extra_provenance=None, validate=True, **kwargs):
        self.path = path
        self.mode = mode

        if mode not in ["r", "w"]:
            raise ValueError(f"File 'mode' argument must be 'r' or 'w' not '{mode}'")

        self.file = self.open(path, mode, **kwargs)

    @classmethod
    def open(cls, path, mode):
        """
        Open a data file.  The base implementation of this function just
        opens and returns a standard python file object.

        Subclasses can override to either open files using different openers
        (like fitsio.FITS), or, for more specific data types, return an
        instance of the class itself to use as an intermediary for the file.

        """
        return open(path, mode)

    @classmethod
    def read(cls, path):
        """ Reads a data file """
        raise NotImplementedError()

    @classmethod
    def write(cls, data, path, **kwargs):
        """ Write a data file """
        raise NotImplementedError()

    @classmethod
    def make_name(cls, tag):
        """Construct and return file name for a particular data tag """
        if cls.suffix:
            return f"{tag}.{cls.suffix}"
        else:
            return tag



class DataStore(dict):
    """Class to provide a transient data store

    This class:
    1) associates data products with keys
    2) provides functions to read and write the various data produces to associated files
    """
    allow_overwrite = False

    def __init__(self, **kwargs):
        """ Build from keywords

        Note
        ----
        All of the values must be data handles of this will raise a TypeError
        """
        dict.__init__(self)
        for key, val in kwargs.items():
            self[key] = val

    def __str__(self):
        """ Override __str__ casting to deal with `TableHandle` objects in the map """
        s = "{"
        for key, val in self.items():
            s += f"  {key}:{val}\n"
        s += "}"
        return s

    def __repr__(self):
        """ A custom representation """
        s = "DataStore\n"
        s += self.__str__()
        return s

    def __setitem__(self, key, value):
        """ Override the __setitem__ to work with `TableHandle` """
        if not isinstance(value, DataHandle):
            raise TypeError(f"Can only add objects of type DataHandle to DataStore, not {type(value)}")
        check = self.get(key)
        if check is not None and not self.allow_overwrite:
            raise ValueError(f"DataStore already has an item with key {key}, of type {type(check)}, created by {check.creator}")
        dict.__setitem__(self, key, value)
        return value

    def __getattr__(self, key):
        """ Allow attribute-like parameter access """
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        """ Allow attribute-like parameter setting """
        return self.__setitem__(key, value)

    def add_data(self, key, data, handle_class, path=None, creator='DataStore'):
        """ Create a handle for some data, and insert it into the DataStore """
        handle = handle_class(key, path=path, data=data, creator=creator)
        self[key] = handle
        return handle

    def read_file(self, key, handle_class, path, creator='DataStore', **kwargs):
        """ Create a handle, use it to read a file, and insert it into the DataStore """
        handle = handle_class(key, path=path, data=None, creator=creator)
        handle.read(**kwargs)
        self[key] = handle
        return handle

    def read(self, key, force=False, **kwargs):
        """ Read the data associated to a particular key """
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to read data {key} because {msg}") from msg
        return handle.read(force, **kwargs)

    def open(self, key, mode='r', **kwargs):
        """ Open and return the file associated to a particular key """
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to open data {key} because {msg}") from msg
        return handle.open(mode, **kwargs)

    def write(self, key, **kwargs):
        """ Write the data associated to a particular key """
        try:
            handle = self[key]
        except KeyError as msg:
            raise KeyError(f"Failed to write data {key} because {msg}") from msg
        return handle.write(**kwargs)

    def write_all(self, force=False, **kwargs):
        """ Write all the data in this DataStore """
        for key, handle in self.items():
            local_kwargs = kwargs.get(key, {})
            if handle.is_written and not force:
                continue
            handle.write(**local_kwargs)



_DATA_STORE = DataStore()

def DATA_STORE():
    """Return the factory instance"""
    return _DATA_STORE
