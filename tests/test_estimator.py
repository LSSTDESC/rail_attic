import pytest
import os
import numpy as np
from rail.estimation.estimator import Estimator
from rail.estimation.utils import write_output_file

# this is temporary until unit test uses a definite test data set and creates
# the yaml file on the fly

test_base_yaml = 'tests/base.yaml'


def test_initialization():
    # test handling of an inexistent config input file
    with pytest.raises(FileNotFoundError):
        instance = Estimator(base_config='non_existent.yaml')

    # assert correct instantiation based on a yaml file
    instance = Estimator(base_config=test_base_yaml)

def test_factory():
    Estimator._subclasses = {}
    for name in ['randomPZ', 'simpleNN']:
        code = Estimator._find_subclass(name)
    assert len(Estimator._subclasses)==2
    
def test_loading():
    assert True

def test_train_not_implemented():
    with pytest.raises(NotImplementedError):
        instance = Estimator(base_config=test_base_yaml)
        instance.inform()

def test_estimate_not_implemented():
    fake_data = {'u': 99., 'g': 99., 'r': 99.}
    with pytest.raises(NotImplementedError):
        instance = Estimator(base_config=test_base_yaml)
        instance.estimate(fake_data)
        
def test_writing(tmpdir):
    instance = Estimator(test_base_yaml)
    instance.zmode = 0
    instance.zgrid = np.arange(0, 1, 0.2)
    instance.pz_pdf = np.ones(5)
    instance.saveloc = tmpdir.join("test.hdf5")
    instance.nzbins = len(instance.zgrid)
    test_dict = {'zmode': instance.zmode, 'pz_pdf': instance.pz_pdf}
    write_output_file(instance.saveloc, instance.num_rows,
                      instance.nzbins, test_dict, instance.zgrid)

    assert os.path.exists(instance.saveloc)
