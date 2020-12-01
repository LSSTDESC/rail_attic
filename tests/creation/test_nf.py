import pytest
import inspect
import os
from rail.creation import normalizingFlow as nf
from rail.creation.rqNeuralSplineFlow import RQNeuralSplineFlow as rqnsf
import dill
import flax

nf_path=inspect.getfile(nf)
saved_file=os.path.join(os.path.dirname(nf_path),'demo_flow.pkl')

#test that module is provided
def test_init1():
    with pytest.raises(ValueError):
        instance=nf.NormalizingFlow(None,file=saved_file)
#test that file or hyperparams must be provided
def test_init2():
    with pytest.raises(ValueError):
        instance=nf.NormalizingFlow(rqnsf,hyperparams=None,file=None)
#test that nfeatures is present in the hyperparams
def test_init3():        
    with pytest.raises(KeyError):
        with open(saved_file, 'rb') as handle:
            save_dict = dill.load(handle)
            save_dict.pop('nfeatures')
            save_dict.pickle('failing_file.pkl')
            instance=nf.NormalizingFlow(rqnsf,file='failing_file.pkl')
#test that flax wrappers are created
def test_init4():
    instance=nf.NormalizingFlow(rqnsf,file=saved_file)
    for func in ['_forward', '_inverse', '_sampler','_log_prob']:
        assert isinstance(getattr(instance, func),flax.nn.base.Model)
#test that params are correctly generated if not provided
def test_init5():
    with open(saved_file, 'rb') as handle:
        save_dict = dill.load(handle)
        instance=nf.NormalizingFlow(rqnsf,save_dict['hyperparams'])
        assert (instance.params is not None)
    
