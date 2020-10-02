import pytest
import numpy as np
from rail.estimation.estimator import Estimator
from rail.estimation.utils import *

test_base_yaml = './tests/base.yaml'

def test_factory():
    for name in ['randomPZ', 'simpleNN']:
        code = Estimator._find_subclass(name)
    assert len(Estimator._subclasses)==2
    with pytest.raises(KeyError) as e:
        non_existant = Estimator._find_subclass('non_existant')
    
def one_algo(single_estimator, single_input):
    """
    A basic test of simpleNN subclass
    """
    
    pz = single_estimator(test_base_yaml,single_input)
    pz.inform()
    for _, end, data in iter_chunk_hdf5_data(pz.testfile, pz._chunk_size,
                                             pz.hdf5_groupname):
        pz_dict = pz.estimate(data)
    assert end == pz.num_rows
    xinputs = single_input['run_params']
    assert len(pz.zgrid) == np.int32(xinputs['nzbins'])

def test_random_pz():
    pz_dict = {'run_params': {'rand_width': 0.025, 'rand_zmin': 0.0, 
                              'rand_zmax': 3.0, 'nzbins': 301}}
    pz_algo = Estimator._find_subclass('randomPZ')
    one_algo(pz_algo,pz_dict)

def test_simple_nn():
    pz_dict = {'run_params': {'width': 0.025, 'zmin': 0.0, 'zmax': 3.0,
                    'nzbins': 301}}
    pz_algo = Estimator._find_subclass('simpleNN')
    one_algo(pz_algo,pz_dict)

def test_flexzboost():
    pz_dict = {'run_params': {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301, 
                              'trainfrac': 0.75, 'bumpmin': 0.02, 'bumpmax': 0.35,
                              'nbump': 3, 'sharpmin': 0.7, 'sharpmax': 2.1,
                              'nsharp': 3, 'max_basis': 35, 
                              'basis_system': 'cosine',
                              'regression_params': {'max_depth': 8,
                                                    'objective':'reg:squarederror'}
                              }}
    pz_algo = Estimator._find_subclass('FZBoost')
    one_algo(pz_algo,pz_dict)
