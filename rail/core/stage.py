""" Base class for PipelineStages in Rail """

import os

from ceci import PipelineStage
from ceci.config import StageParameter as Param
from rail.core.data import DATA_STORE, DataHandle


class RailStage(PipelineStage):
    """Base class for rail stages

    This inherits from `ceci.PipelineStage` and implements rail-specific data handling
    In particular, this provides some very useful features:

    1.  Access to the `DataStore`, which keeps track of the various data used in a pipeline, and
    provides access to each by a unique key.

    2.  Functionality to help manage multiple instances of a particular class of stage.
    The original ceci design didn't have a mechanism to handle this.  If you tried
    you would run into name clashes between the different instances.  In `ceci` 1.7 we
    added functionality to `ceci` to allow you to have multiple instances of a single class,
    in particular we distinguish between the class name (`cls.name`) and and the name of
    the particular instance (`self.instance_name`) and added aliasing for inputs and outputs,
    so that different instances of `PipelineStage` would be able to give different names
    to their inputs and outputs.  However, using that functionality in a consistent way
    requires a bit of care.  So here we are providing methods to do that, and to do it in
    a way that uses the `DataStore` to keep track of the various data products.

    Notes
    -----
    These methods typically take a tag as input (i.e., something like "input"),
    but use the "aliased_tag" (i.e., something like "inform_pz_input") when interacting
    with the DataStore.

    In particular, the `get_handle()`, `get_data()` and `input_iterator()` will get the data
    from the DataStore under the aliased tag.  E.g., if you call `self.get_data('input')` for
    a `Stage` that has aliased "input" to "special_pz_input", it will
    get the data associated to "special_pz_input" in the DataStore.

    Similarly, `add_handle()` and `set_data()` will add the data to the DataStore under the aliased tag
    e.g., if you call `self.set_data('input')` for a `Stage` that has
    aliased "input" to "special_pz_input", it will store the data in the DataStore
    under the key "special_pz_input".

    And `connect_input()` will do the alias lookup both on the input and output.
    I.e., it is the same as calling
    `self.set_data(inputTag, other.get_handle(outputTag, allow_missing=True), do_read=False)`
    """

    config_options = dict(output_mode=Param(str, 'default',
                                            msg="What to do with the outputs"))

    data_store = DATA_STORE()

    def __init__(self, args, comm=None):
        """ Constructor:
        Do RailStage specific initialization """
        PipelineStage.__init__(self, args, comm=comm)
        self._input_length = None

    def get_handle(self, tag, path=None, allow_missing=False):
        """Gets a DataHandle associated to a particular tag

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
        path : str or None
            The path to the data, only needed if we might need to read the data
        allow_missing : bool
            If False this will raise a key error if the tag is not in the DataStore

        Returns
        -------
        handle : DataHandle
            The handle that give access to the associated data
        """
        aliased_tag = self.get_aliased_tag(tag)
        handle = self.data_store.get(aliased_tag)
        if handle is None:
            if not allow_missing:
                raise KeyError(f'{self.instance_name} failed to get data by handle {aliased_tag}, associated to {tag}')
            handle = self.add_handle(tag, path=path)
        return handle

    def add_handle(self, tag, data=None, path=None):
        """Adds a DataHandle associated to a particular tag

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
        data : any or None
            If not None these data will be associated to the handle
        path : str or None
            If not None, this will be the path used to read the data

        Returns
        -------
        handle : DataHandle
            The handle that gives access to the associated data
        """
        aliased_tag = self.get_aliased_tag(tag)
        if aliased_tag in self._inputs:
            if path is None:
                path = self.get_input(aliased_tag)
            handle_type = self.get_input_type(tag)
        else:
            if path is None:
                path = self.get_output(aliased_tag)
            handle_type = self.get_output_type(tag)
        handle = handle_type(aliased_tag, path=path, data=data, creator=self.instance_name)
        print(f"Inserting handle into data store.  {aliased_tag}: {handle.path}, {handle.creator}")
        self.data_store[aliased_tag] = handle
        return handle

    def get_data(self, tag, allow_missing=True):
        """Gets the data associated to a particular tag

        Notes
        -----
        1. This gets the data via the DataHandle, and can and will read the data
        from disk if needed.

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
        allow_missing : bool
            If False this will raise a key error if the tag is not in the DataStore

        Returns
        -------
        data : any
            The data accesed by the handle assocated to the tag
        """
        handle = self.get_handle(tag, allow_missing=allow_missing)
        if not handle.has_data:
            handle.read()
        return handle()

    def set_data(self, tag, data, path=None, do_read=True):
        """Sets the data associated to a particular tag

        Notes
        -----
        1. If data is a DataHandle and tag is one of the input tags,
        then this will add an alias between the two, i.e., it will
        set `self.config.alias[tag] = data.tag`.  This allows the user to
        make connections between stages simply by passing DataHandles between
        them.

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
        data : any
            The data being set,
        path : str or None
            Can be used to set the path for the data
        do_read : bool
            If True, will read the data if it is not set

        Returns
        -------
        data : any
            The data accesed by the handle assocated to the tag
        """
        if isinstance(data, DataHandle):
            aliased_tag = data.tag
            if tag in self.input_tags():
                self.config.aliases[tag] = aliased_tag
                if data.has_path:
                    self._inputs[tag] = data.path
            arg_data = data.data
        else:
            if path is None:
                arg_data = data
            else:
                arg_data = None

        handle = self.get_handle(tag, path=path, allow_missing=True)
        if not handle.has_data:
            if arg_data is None and do_read:
                handle.read()
            if arg_data is not None:
                handle.data = arg_data
        return handle.data

    def add_data(self, tag, data=None):
        """Adds a handle to the DataStore associated to a particular tag and
        attaches data to it.

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
        data : any

        Returns
        -------
        data : any
            The data accesed by the handle assocated to the tag
        """
        handle = self.add_handle(tag, data=data)
        return handle.data

    def input_iterator(self, tag, **kwargs):
        """Iterate the input assocated to a particular tag

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data

        Keywords
        --------
        These will be passed to the Handle's iterator method
        """
        handle = self.get_handle(tag, allow_missing=True)
        if not handle.has_data:  #pragma: no cover
            handle.read()
        if self.config.hdf5_groupname:
            self._input_length = handle.size(groupname=self.config.hdf5_groupname)
            kwcopy = dict(groupname=self.config.hdf5_groupname,
                          chunk_size=self.config.chunk_size,
                          rank=self.rank,
                          parallel_size=self.size)
            kwcopy.update(**kwargs)
            return handle.iterator(**kwcopy)
        else:  #pragma:  no cover
            test_data = self.get_data('input')
            s = 0
            e = len(list(test_data.items())[0][1])
            self._input_length=e
            iterator=[[s, e, test_data]]
            return iterator

    def connect_input(self, other, inputTag=None, outputTag=None):
        """Connect another stage to this stage as an input

        Parameters
        ----------
        other : RailStage
             The stage whose output is being connected
        inputTag : str
             Which input tag of this stage to connect to.  None -> self.inputs[0]
        outputTag : str
             Which output tag of the other stage to connect to.  None -> other.outputs[0]

        Returns
        -------
        handle : The input handle for this stage
        """
        if inputTag is None:
            inputTag = self.inputs[0][0]  #pylint: disable=no-member
        if outputTag is None:
            outputTag = other.outputs[0][0]
        handle = other.get_handle(outputTag, allow_missing=True)
        return self.set_data(inputTag, handle, do_read=False)

    def _finalize_tag(self, tag):
        """Finalize the data for a particular tag.

        This can be overridden by sub-classes for more complicated behavior
        """
        handle = self.get_handle(tag, allow_missing=True)
        if self.config.output_mode == 'default':
            if not os.path.exists(handle.path):
                handle.write()
        final_name = PipelineStage._finalize_tag(self, tag)
        handle.path = final_name
        return final_name
