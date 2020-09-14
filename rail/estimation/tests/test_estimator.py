import pytest
import os
import numpy as np
from rail.estimation.estimator import Estimator

#this is temporary until unit test uses a definite test data set and creates the yaml file on the fly
import inspect
import rail
test_base_yaml =  os.path.join(os.path.dirname(inspect.getfile(rail)),'estimation/base.yaml') 

def test_initialization():
    
    #test handling of an inexistent config input file
    with pytest.raises(FileNotFoundError) as e_info:
        instance = Estimator(base_config='non_existent.yaml')
        
    #assert correct instantiation based on a yaml file 
    instance = Estimator(base_config=test_base_yaml)

def test_loading():
    assert True

def test_writing(tmpdir):
    instance = Estimator(test_base_yaml)
    
    # with pytest.raises(AttributeError) as e_info:
    #     for attr in ['zmode','zgrid','pz_pdf']:
    #         getattr(instance,attr)
    instance.zmode = 0
    instance.zgrid = np.arange(0,1,0.2)
    instance.pz_pdf = np.ones(5)
    instance.saveloc=tmpdir.join("test.hdf5")
    instance.write_out()
    assert os.path.exists(instance.saveloc)

def test_randompz(tmpdir):
    from rail.estimation.algos import random
    inputs = {'run_params':{'rand_width':0.025,'rand_zmin':0.0, 'rand_zmax':3.0,'rand_zbins':301}}

    instance = random.randomPZ(base_config=test_base_yaml, config_dict=inputs)
    #assert correct loading of the config
    assert instance.width == inputs['run_params']['rand_width']
    assert instance.zmin == inputs['run_params']['rand_zmin']
    assert instance.zmax == inputs['run_params']['rand_zmax']
    assert instance.nzbins == inputs['run_params']['rand_zbins']

    instance.run_photoz()
    #assert correct execution of run_photoz
    assert instance.zmode is not None
    assert len(instance.zmode) != 0
    assert len(instance.zgrid) == instance.nzbins
    assert instance.pz_pdf is not None
    assert len(instance.pz_pdf[0]) == instance.nzbins
