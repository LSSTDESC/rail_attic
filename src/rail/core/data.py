"""Rail-specific data management"""

import os
import tables_io
import pickle
import qp

from descformats import base, handle, data


class DataStore(data.DataStore):
    pass

class DataHandle(handle.DataHandle):
    pass
    
class TableHandle(base.TableHandle):
    pass

class Hdf5Handle(base.Hdf5Handle):
    pass

class FitsHandle(base.FitsHandle):
    pass

class PqHandle(base.PqHandle):
    pass

class QPHandle(base.QPHandle):
    pass


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
    def _open(cls, path, mode, **kwargs):
        """Open and return the associated file
        """
        if mode == 'w':
            return cls.model_factory.open(path, mode='wb', **kwargs)
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
    def _open(cls, path, mode, **kwargs):  #pylint: disable=unused-argument
        if mode == 'w':  #pragma: no cover
            raise NotImplementedError("Use FlowHandle.write(), not FlowHandle.open(mode='w')")
        return cls.flow_factory.read(path)

    @classmethod
    def _read(cls, path, **kwargs):
        """Read and return the data from the associated file """
        return cls.flow_factory.read(path, **kwargs)

    @classmethod
    def _write(cls, data, path, **kwargs):
        return data.save(path)


