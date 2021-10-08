import pytest
import os
import numpy as np
from rail.estimation.estimator import Estimator
from tables_io.ioUtils import initializeHdf5Write, writeDictToHdf5Chunk
from tables_io.ioUtils import finalizeHdf5Write
import yaml

# this is temporary until unit test uses a definite test data set and creates
# the yaml file on the fly

test_base_yaml = 'tests/base.yaml'


def test_init_with_dict():
    # test we can init with a dict we have already loaded
    d = yaml.safe_load(open(test_base_yaml))['base_config']
    _ = Estimator(d)


def test_initialization():
    # test handling of an inexistent config input file
    with pytest.raises(FileNotFoundError):
        _ = Estimator(base_config='non_existent.yaml')

    # assert correct instantiation based on a yaml file
    _ = Estimator(base_config=test_base_yaml)


def test_loading():
    assert True


def test_train_not_implemented():
    fakedata = {'u': 99., 'g': 99., 'r': 99.}
    with pytest.raises(NotImplementedError):
        instance = Estimator(base_config=test_base_yaml)
        instance.inform(fakedata)


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
    group, fout = initializeHdf5Write(instance.saveloc, 'data',
                                      zmode=((1,), 'f4'),
                                      pz_pdf=((1, 5), 'f4'))
    writeDictToHdf5Chunk(group, test_dict, 0, 1, zmode='zmode',
                         pz_pdf='pz_pdf')
    finalizeHdf5Write(fout, zgrid=instance.zgrid)
    assert os.path.exists(instance.saveloc)


def test_find_subclass():
    _ = Estimator._find_subclass('randomPZ')
