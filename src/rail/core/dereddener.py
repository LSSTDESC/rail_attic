
import tables_io
from astropy.coordinates import SkyCoord

from rail.core.stage import RailStage

from rail.core.data import PqHandle, Hdf5Handle

dustmaps_config = tables_io.lazy_modules.lazyImport('dustmaps.config')
dustmaps_sfd = tables_io.lazy_modules.lazyImport('dustmaps.sfd')


class Dereddener(RailStage):
    """Utility stage that does dereddening

    """
    name = 'Dereddener'

    config_options = RailStage.config_options.copy()
    config_options.update(bands='ugrizy')
    config_options.update(mag_name="mag_{band}_lsst")
    config_options.update(band_a_env=[4.81,3.64,2.70,2.06,1.58,1.31])
    config_options.update(dustmap_name='sfd')
    config_options.update(dustmap_dir=str)
    config_options.update(copy_cols=[])

    inputs = [('input', Hdf5Handle)]
    outputs = [('output', Hdf5Handle)]

    def fetch_map(self):
        dust_map_dict = dict(sfd=dustmaps_sfd)
        try:
            dust_map_submod = dust_map_dict[self.config.dustmap_name]
        except KeyError as msg:  # pragma: no cover
            raise KeyError(f"Unknown dustmap {self.config.dustmap_name}, options are {list(dust_map_dict.keys())}") from msg

        if os.path.exists(os.path.join(self.config.dustmap_dir, self.config.dustmap_name)):  # pragma: no cover
            # already downloaded, return
            return
        
        dust_map_config = dustmaps_config.config
        dust_map_config['data_dir'] = self.config.dustmap_dir
        fetch_func = dust_map_submod.fetch
        fetch_func()
        
            
    def __init__(self, args, comm=None):
        RailStage.__init__(self, args, comm=comm)

    def run(self):
        data = self.get_data('input', allow_missing=True)
        out_data = {}
        coords = SkyCoord(data['ra'], data['decl'], unit = 'deg',frame='fk5')
        dust_map_dict = dict(sfd=dustmaps_sfd.SFDQuery)
        try:
            dust_map_class = dust_map_dict[self.config.dustmap_name]
            dust_map_config = dustmaps_config.config
            dust_map_config['data_dir'] = self.config.dustmap_dir
            dust_map = dust_map_class()
        except KeyError as msg:  # pragma: no cover
            raise KeyError(f"Unknown dustmap {self.config.dustmap_name}, options are {list(dust_map_dict.keys())}") from msg
        ebvvec = dust_map(coords)
        for i, band_ in enumerate(self.config.bands):
            band_mag_name = self.config.mag_name.format(band=band_)
            mag_vals = data[band_mag_name]
            out_data[band_mag_name] = mag_vals - ebvvec*self.config.band_a_env[i]
        for col_ in self.config.copy_cols:  # pragma: no cover
            out_data[col_] = data[col_]
        self.add_data('output', out_data)

    def __call__(self, data):
        """Return a converted table

        Parameters
        ----------
        data : table-like
            The data to be converted

        Returns
        -------
        out_data : table-like
            The converted version of the table
        """
        self.set_data('input', data)
        self.run()
        return self.get_handle('output')
