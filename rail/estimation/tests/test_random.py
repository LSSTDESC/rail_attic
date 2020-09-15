import yaml
import rail
from rail.estimation.estimator import Estimator
from rail.estimation.utils import *

def test_random():
    """
    A couple of basic tests of the random class
    """
    base_yaml = "base.yaml"
    with open(base_yaml, 'r') as f:
        base_dict = yaml.safe_load(f)['base_config']
    input_yaml = "../configs/randomPZ.yaml"
    name = input_yaml.split("/")[-1].split(".")[0]

    with open(input_yaml, 'r') as f:
        config_dict=yaml.safe_load(f)

    code = Estimator._find_subclass(name)
    pz = code(config_dict)

    for start, end, data in iter_chunk_hdf5_data(pz.testfile,pz._chunk_size,
                                                 base_dict['hdf5_groupname']):
        pz_dict = pz.run_photoz(data)
    assert end == pz.num_rows
    #print(len(pz.zgrid))
    #print("how many zbins?")
    xinputs = config_dict['run_params']
    assert len(pz.zgrid) == np.int32(xinputs['nzbins'])

if __name__=="__main__":
    test_random()
