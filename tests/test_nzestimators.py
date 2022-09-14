import os
import pytest
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.estimation.algos import NZDir
from rail.core.utils import RAILDIR


testszdata = os.path.join(RAILDIR, 'rail/examples/testdata/training_100gal.hdf5')
testphotdata = os.path.join(RAILDIR, 'rail/examples/testdata/validation_10gal.hdf5')
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(key, inform_class, estimator_class, summary_kwargs):
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    spec_data = DS.read_file('spec_data', TableHandle, testszdata)
    phot_data = DS.read_file('phot_data', TableHandle, testphotdata)
    informer = inform_class.make_stage(name=f"inform_{key}", model="tmp.pkl")
    informer.inform(spec_data)
    estimatorr = estimator_class.make_stage(name=key, model=informer.get_handle('model'), **summary_kwargs)
    summary_ens = estimatorr.estimate(phot_data)
    os.remove(estimatorr.get_output(estimatorr.get_aliased_tag('output'), final_name=True))
    os.remove("tmp.pkl")
    return summary_ens


def test_NZDir():
    summary_config_dict = {}
    inform_class = NZDir.Inform_NZDir
    estimator_class = NZDir.NZDir
    _ = one_algo("NZDir", inform_class, estimator_class, summary_config_dict)


def test_NZDir_bad_weight():
    summary_config_dict = {'phot_weightcol': 'notarealcol'}
    inform_class = NZDir.Inform_NZDir
    estimator_class = NZDir.NZDir
    with pytest.raises(KeyError):
        _ = one_algo("NZDir", inform_class, estimator_class, summary_config_dict)
