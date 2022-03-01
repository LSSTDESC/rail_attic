""" Base class for PipelineStages in Rail """

from ceci.config import StageParameter as Param
from ceci import PipelineStage

from rail.core.data import DATA_STORE, DataHandle


class RailStage(PipelineStage):
    """ Base class for rail stages

    Implements rail-specific data handling
    """

    config_options = dict(output_mode=Param(str, 'default',
                                            msg="What to do with the outputs"))

    data_store = DATA_STORE()

    def get_handle(self, tag, allow_missing=False):
        """Gets a DataHandle associated to a particular tag

        Note that this will get the data from the DataStore under the aliased tag

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
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
            handle = self.add_handle(tag)
        return handle


    def add_handle(self, tag, data=None):
        """Adds a DataHandle associated to a particular tag

        Note that this will add the data to the DataStore under the aliased tag

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
        data : any or None
            If not None these data will be associated to the handle

        Returns
        -------
        handle : DataHandle
            The handle that gives access to the associated data
        """
        aliased_tag = self.get_aliased_tag(tag)
        if aliased_tag in self._inputs:
            path = self.get_input(aliased_tag)
            handle_type = self.get_input_type(tag)
        else:
            path = self.get_output(aliased_tag)
            handle_type = self.get_output_type(tag)
        handle = handle_type(aliased_tag, path=path, data=data, creator=self.instance_name)
        print(f"Inserting handle into data store.  {aliased_tag}: {handle.path}, {handle.creator}")
        self.data_store[aliased_tag] = handle
        return handle

    def get_data(self, tag, allow_missing=False):
        """Gets the data associated to a particular tag

        Note that this will get the data from the DataStore under the aliased tag

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

        handle = self.get_handle(tag, allow_missing)
        if not handle.has_data:
            handle.read()
        return handle.data

    def set_data(self, tag, data, do_read=True):
        """Sets the data associated to a particular tag

        Note that this will set the data in the handle from the DataStore
        under the aliased tag.

        Parameters
        ----------
        tag : str
            The tag (from cls.inputs or cls.outputs) for this data
        data : any
            The data being set
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
            arg_data = data

        handle = self.get_handle(tag, allow_missing=True)
        if not handle.has_data:
            if arg_data is None and do_read:
                handle.read()
            handle.data = arg_data
        return handle.data

    def add_data(self, tag, data=None):
        """Adds data and an handle to the DataStore associated to a particular tag

        Note that this will set the data in the handle from the DataStore
        under the aliased tag.

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
        These will be passed to the Handle's iterate function
        """
        handle = self.get_handle(tag)
        kwcopy = dict(groupname=self.config.hdf5_groupname,
                      chunk_size=self.config.chunk_size)
        kwcopy.update(**kwargs)
        return handle.iterator(**kwcopy)

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
        handle = self.get_handle(tag)
        if self.config.output_mode == 'default':
            handle.write()
        final_name = PipelineStage._finalize_tag(self, tag)
        handle.path = final_name
        return final_name
