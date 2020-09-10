import pytest
from rail.estimation.estimator import Estimator

def test_initialization():
    
    #test handling of an inexistent config input file
    with pytest.raises(FileNotFoundError) as e_info:
        instance = Estimator('non_existent.yaml')
        
    #assert correct instantiation based on a yaml file 
    instance = Estimator('base.yaml')

    #assert correct instantiation based on a dictionary
    instance = Estimator(base_config)
    
def test_loading():
    assert True

def test_writing():
    instance = Estimator('base.yaml')
    
    with pytest.raises(AttributeError) as e_info:
        for attr in ['zmode','zgrid','pz_pdf']:
            getattr(instance,attr)
    
    instance.saveloc="test.hdf5"
    instance.write_out()
    assert os.path.exists("test.hdf5")
    os.system("rm test.hdf5")
