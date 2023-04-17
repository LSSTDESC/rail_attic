"""Rail-specific data management"""

import os
import tables_io
import pickle
import qp



class DataHandle:
    """Class to act as a handle for a bit of data.  Associating it with a file and
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
        self.fileObj = None
        self.groups = None
        self.partial = False

    def open(self, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        if self.path is None:
            raise ValueError("DataHandle.open() called but path has not been specified")
        self.fileObj = self._open(self.path, **kwargs)
        return self.fileObj

    @classmethod
    def _open(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._open")  #pragma: no cover

    def close(self, **kwargs):  #pylint: disable=unused-argument
        """Close """
        self.fileObj = None

    def read(self, force=False, **kwargs):
        """Read and return the data from the associated file """
        if self.data is not None and not force:
            return self.data
        self.set_data(self._read(self.path, **kwargs))
        return self.data

    def __call__(self, **kwargs):
        """Return the data, re-reading the fill if needed"""
        if self.has_data and not self.partial:
            return self.data
        return self.read(force=True, **kwargs)

    @classmethod
    def _read(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._read")  #pragma: no cover

    def write(self, **kwargs):
        """Write the data to the associatied file """
        if self.path is None:
            raise ValueError("TableHandle.write() called but path has not been specified")
        if self.data is None:
            raise ValueError(f"TableHandle.write() called for path {self.path} with no data")
        outdir = os.path.dirname(os.path.abspath(self.path))
        if not os.path.exists(outdir):  #pragma: no cover
            os.makedirs(outdir, exist_ok=True)
        return self._write(self.data, self.path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        raise NotImplementedError("DataHandle._write")  #pragma: no cover

    def initialize_write(self, data_lenght, **kwargs):
        """Initialize file to be written by chunks"""
        if self.path is None:  #pragma: no cover
            raise ValueError("TableHandle.write() called but path has not been specified")
        self.groups, self.fileObj = self._initialize_write(self.data, self.path, data_lenght, **kwargs)

    @classmethod
    def _initialize_write(cls, data, path, data_lenght, **kwargs):
        raise NotImplementedError("DataHandle._initialize_write") #pragma: no cover

    def write_chunk(self, start, end, **kwargs):
        """Write the data to the associatied file """
        if self.data is None:
            raise ValueError(f"TableHandle.write_chunk() called for path {self.path} with no data")
        if self.fileObj is None:
            raise ValueError(f"TableHandle.write_chunk() called before open for {self.tag} : {self.path}")
        return self._write_chunk(self.data, self.fileObj, self.groups, start, end, **kwargs)


    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        raise NotImplementedError("DataHandle._write_chunk")  #pragma: no cover

    def finalize_write(self, **kwargs):
        """Finalize and close file written by chunks"""
        if self.fileObj is None:  #pragma: no cover
            raise ValueError(f"TableHandle.finalize_wite() called before open for {self.tag} : {self.path}")
        self._finalize_write(self.data, self.fileObj, **kwargs)

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        raise NotImplementedError("DataHandle._finalize_write")  #pragma: no cover

    def iterator(self, **kwargs):
        """Iterator over the data"""
        #if self.data is not None:
        #    for i in range(1):
        #        yield i, -1, self.data
        return self._iterator(self.path, **kwargs)

    def set_data(self, data, partial=False):
        """Set the data for a chunk, and set the partial flag to true"""
        self.data = data
        self.partial = partial

    def size(self, **kwargs):
        """Return the size of the data associated to this handle"""
        return self._size(self.path, **kwargs)

    @classmethod
    def _size(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._size")  #pragma: no cover

    @classmethod
    def _iterator(cls, path, **kwargs):
        raise NotImplementedError("DataHandle._iterator")  #pragma: no cover

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
            return tag  #pragma: no cover


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
        return tables_io.io.io_open(path, **kwargs)  #pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return tables_io.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associatied file """
        return tables_io.write(data, path, **kwargs)

    @classmethod
    def _size(cls, path, **kwargs):
        return tables_io.io.getInputDataLengthHdf5(path, **kwargs)

    @classmethod
    def _iterator(cls, path, **kwargs):
        """Iterate over the data"""
        return tables_io.iteratorNative(path, **kwargs)

class Hdf5Handle(TableHandle):
    """DataHandle for a table written to HDF5"""
    suffix = 'hdf5'

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        if groups is None:
            tables_io.io.writeDictToHdf5ChunkSingle(fileObj, data, start, end, **kwargs)
        else:  #pragma: no cover
            tables_io.io.writeDictToHdf5Chunk(groups, data, start, end, **kwargs)


class FitsHandle(TableHandle):
    """DataHandle for a table written to fits"""
    suffix = 'fits'


class PqHandle(TableHandle):
    """DataHandle for a parquet table"""
    suffix = 'pq'


class QPHandle(DataHandle):
    """DataHandle for qp ensembles
    """
    suffix = 'hdf5'

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file

        Notes
        -----
        This will simply open the file and return a file-like object to the caller.
        It will not read or cache the data
        """
        return tables_io.io.io_open(path, **kwargs)  #pylint: disable=no-member

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return qp.read(path)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associatied file """
        return data.write_to(path)

    @classmethod
    def _initialize_write(cls, data, path, data_lenght, **kwargs):
        comm = kwargs.get('communicator', None)
        return data.initializeHdf5Write(path, data_lenght, comm)

    @classmethod
    def _write_chunk(cls, data, fileObj, groups, start, end, **kwargs):
        return data.writeHdf5Chunk(fileObj, start, end)

    @classmethod
    def _finalize_write(cls, data, fileObj, **kwargs):
        return data.finalizeHdf5Write(fileObj)


def default_model_read(modelfile):
    """Default function to read model files, simply used pickle.load"""
    return pickle.load(open(modelfile, 'rb'))


def default_model_write(model, path):
    """Write the model, this default implementation uses pickle"""
    with open(path, 'wb') as fout:
        pickle.dump(obj=model, file=fout, protocol=pickle.HIGHEST_PROTOCOL)


class ModelDict(dict):
    """
    A specialized dict to keep track of individual estimation models objects: this is just a dict these additional features

    1. Keys are paths
    2. There is a read(path, force=False) method that reads a model object and inserts it into the dictionary
    3. There is a single static instance of this class
    """
    def open(self, path, mode, **kwargs):  #pylint: disable=no-self-use
        """Open the file and return the file handle"""
        return open(path, mode, **kwargs)

    def read(self, path, force=False, reader=None, **kwargs):  #pylint: disable=unused-argument
        """Read a model into this dict"""
        if reader is None:
            reader = default_model_read
        if force or path not in self:
            model = reader(path)
            self.__setitem__(path, model)
            return model
        return self[path]

    def write(self, model, path, force=False, writer=None, **kwargs):  #pylint: disable=unused-argument
        """Write the model, this default implementation uses pickle"""
        if writer is None:
            writer = default_model_write
        if force or path not in self:
            self.__setitem__(path, model)
            writer(model, path)



class ModelHandle(DataHandle):
    """DataHandle for machine learning models
    """
    suffix = 'pkl'

    model_factory = ModelDict()

    @classmethod
    def _open(cls, path, **kwargs):
        """Open and return the associated file
        """
        kwcopy = kwargs.copy()
        if kwcopy.pop('mode', 'r') == 'w':
            return cls.model_factory.open(path, mode='wb', **kwcopy)
        return cls.model_factory.read(path, **kwargs)

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return cls.model_factory.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        """Write the data to the associatied file """
        return cls.model_factory.write(data, path, **kwargs)



class FlowDict(dict):
    """
    A specialized dict to keep track of individual flow objects: this is just a dict these additional features

    1. Keys are paths
    2. Values are flow objects, this is checked at runtime.
    3. There is a read(path, force=False) method that reads a flow object and inserts it into the dictionary
    4. There is a single static instance of this class
    """

    def __setitem__(self, key, value):
        """ Add a key-value pair, and check to make sure that the value is a `Flow` object """
        from pzflow import Flow
        if not isinstance(value, Flow):  #pragma: no cover
            raise TypeError(f"Only values of type Flow can be added to a FlowFactory, not {type(value)}")
        return dict.__setitem__(self, key, value)

    def read(self, path, force=False):
        """ Read a `Flow` object from disk and add it to this dictionary """
        from pzflow import Flow
        if force or path not in self:
            flow = Flow(file=path)
            self.__setitem__(path, flow)
            return flow
        return self[path]  #pragma: no cover


class FlowHandle(ModelHandle):
    """
    A wrapper around a file that describes a PZFlow object
    """
    flow_factory = FlowDict()

    suffix = 'pkl'

    @classmethod
    def _open(cls, path, **kwargs):  #pylint: disable=unused-argument
        if kwargs.get('mode', 'r') == 'w':  #pragma: no cover
            raise NotImplementedError("Use FlowHandle.write(), not FlowHandle.open(mode='w')")
        return cls.flow_factory.read(path)

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return cls.flow_factory.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        return data.save(path)


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
        try:
            return self.__getitem__(key)
        except KeyError as msg:
            # Kludge to get docstrings to work
            if key in ['__objclass__']:  #pragma: no cover
                return None
            raise KeyError from msg

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
        return handle.open(mode=mode, **kwargs)

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
