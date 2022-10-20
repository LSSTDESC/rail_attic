""" Parameters that are shared between stages """

from ceci.config import StageParameter as Param
from ceci.config import StageConfig

SHARED_PARAMS = StageConfig(
    hdf5_groupname=Param(str, "photometry", msg="name of hdf5 group for data, if None, then set to ''"),
    zmin=Param(float, 0.0, msg="The minimum redshift of the z grid"),
    zmax=Param(float, 3.0, msg="The maximum redshift of the z grid"),
    nzbins=Param(int, 301, msg="The number of gridpoints in the z grid"),
    dz=Param(float, 0.01, msg="delta z in grid"),
    nondetect_val=Param(float, 99.0, msg="value to be replaced with magnitude limit for non detects"),
    ref_band=Param(str, "mag_i_lsst", msg="band to use in addition to colors"),
    seed=Param(int, 87, msg="Random number seed"),
    redshift_column_name=Param(str, 'redshift', msg="name of redshift column")
)


def copy_param(param_name):
    """Return a copy of one of the shared parameters"""
    return SHARED_PARAMS.get(param_name).copy()


def set_param_default(param_name, default_value):
    """Change the default value of one of the shared parameters"""
    SHARED_PARAMS.get(param_name).set_default(default_value)
