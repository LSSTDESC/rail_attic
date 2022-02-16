""" Base class for PipelineStages in Rail """

from ceci import PipelineStage

from rail.core.data import DATA_STORE, DataHandle


class RailStage(PipelineStage):
    """ Base class for rail stages

    Implements rail-specific data handling
    """

    config_options = {}

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
        if handle is None:
            return None
        if not handle.has_data:
            handle.read()
        return handle.data

    def set_data(self, tag, data):
        """Sets the data associated to a particular tag

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
        if isinstance(data, DataHandle):
            aliased_tag = data.tag
            if tag in self.input_tags():
                self.config.aliases[tag] = aliased_tag
            arg_data = data.data
        else:
            arg_data = data

        handle = self.get_handle(tag, allow_missing=True)
        if handle is None:
            return None
        if not handle.has_data:
            if arg_data is None:
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
