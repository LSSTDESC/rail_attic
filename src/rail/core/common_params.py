""" Parameters that are shared between stages """

from ceci.config import StageParameter as Param
from ceci.config import StageConfig

lsst_bands = 'ugrizy'
lsst_mag_cols = [f'mag_{band}_lsst' for band in lsst_bands]
lsst_mag_err_cols = [f'mag_err_{band}_lsst' for band in lsst_bands]
lsst_def_maglims = dict(
    mag_u_lsst=27.79,
    mag_g_lsst=29.04,
    mag_r_lsst=29.06,
    mag_i_lsst=28.62,
    mag_z_lsst=27.98,
    mag_y_lsst=27.05
)


SHARED_PARAMS = StageConfig(
    hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"),
    zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
    zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
    nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
    dz=Param(float, 0.01, msg="delta z in grid"),
    nondetect_val=Param(float, 99.0, msg="value to be replaced with magnitude limit for non detects"),
    bands=Param(list, lsst_mag_cols, msg="Names of columns for magnitgude by filter band"),
    err_bands=Param(list, lsst_mag_err_cols, msg="Names of columns for magnitgude errors by filter band"),
    mag_limits=Param(dict, lsst_def_maglims, msg="Limiting magnitdues by filter"),
    ref_band=Param(str, "mag_i_lsst", msg="band to use in addition to colors"),
    redshift_col=Param(str, 'redshift', msg="name of redshift column")
)


def copy_param(param_name):
    """Return a copy of one of the shared parameters"""
    return SHARED_PARAMS.get(param_name).copy()


def set_param_default(param_name, default_value):
    """Change the default value of one of the shared parameters"""
    try:
        SHARED_PARAMS.get(param_name).set_default(default_value)
    except AttributeError as msg:  # pragma: no cover
        raise KeyError(f"No shared parameter {param_name} in SHARED_PARAMS")

def set_param_defaults(**kwargs):  # pragma: no cover
    """Change the default value of several of the shared parameters"""
    for key, val in kwargs.items():
        set_param_default(key, val)

