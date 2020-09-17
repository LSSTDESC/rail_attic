import yaml
import rail
from rail.estimation.estimator import Estimator
from rail.estimation.utils import *
# this is temporary until unit test uses a definite test data set and creates the
# yaml file on the fly
import inspect
import rail
from rail.estimation.algos import randomPZ

os.chdir(os.path.join(os.path.dirname(inspect.getfile(rail)),'estimation/tests/data') )
test_base_yaml =  './base.yaml'

def test_random():
    """
    A couple of basic tests of the random class
    """

    inputs = {'run_params':{'rand_width':0.025,'rand_zmin':0.0, 'rand_zmax':3.0,
                            'nzbins':301}}
    name = 'randomPZ'


    #code = Estimator._find_subclass(name)
    pz = randomPZ.randomPZ(test_base_yaml, inputs)
    #pz = code(test_base_yaml, inputs)
    for start, end, data in iter_chunk_hdf5_data(pz.testfile,pz._chunk_size, pz.hdf5_groupname):
        pz_dict = pz.run_photoz(data)
    assert end == pz.num_rows
    xinputs = inputs['run_params']
    assert len(pz.zgrid) == np.int32(xinputs['nzbins'])

if __name__=="__main__":
    test_random()
