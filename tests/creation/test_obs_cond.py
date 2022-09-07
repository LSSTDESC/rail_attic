# Temporary test file for obs_cond,
# to be merged to test_degraders.py when
# ready.

import os
from typing import Type

import numpy as np
import pandas as pd
import pytest
from rail.core.data import DATA_STORE, TableHandle
from rail.core.utilStages import ColumnMapper
from rail.creation.degradation import *

# Here copied from test_degraders to
# generate data:
@pytest.fixture
def data():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(100, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data", df, TableHandle, path="dummy.pd")


# This is an example to see if the function
# returns the correct shape
def test_ObsCondition_returns_correct_shape(data):
    """Test that the ObsCond returns the correct shape"""

    degrader = ObsCondition.make_stage()

    degraded_data = degrader(data).data

    assert degraded_data.shape == (data.data.shape[0], 2 * data.data.shape[1])
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))
    

# We can further test some functions e.g.
# random seeds
def test_ObsCondition_random_seed(data):
    """Test control with random seeds."""
    degrader1 = ObsCondition.make_stage(random_seed=0)
    degrader2 = ObsCondition.make_stage(random_seed=0)
    
    # make sure setting the same seeds yields the same output
    degraded_data1 = degrader1(data).data
    degraded_data2 = degrader2(data).data
    assert degraded_data1.equals(degraded_data2)

    # make sure setting different seeds yields different output
    degrader3 = ObsCondition.make_stage(random_seed=1)
    degraded_data3 = degrader3(data).data.to_numpy()
    assert not degraded_data1.equals(degraded_data3)
    
    os.remove(degrader3.get_output(degrader3.get_aliased_tag("output"), final_name=True))


# Test for ValueError or TypeError
# for this need a list of config files
# maybe save these in ../data/obs_configs/
# Notes: do not need to test LSSTErrorModel parameters,
# only ObsConditions specific + file directories
@pytest.mark.parametrize(
    "configs,error",
    [
        # band-dependent
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/m5_xx.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/m5_b_False.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/m5_b_xx.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/nVisYr_xx.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/gamma_xx.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/msky_xx.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/theta_b_False.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/km_b_False.yml"), TypeError),
        
        # band-independent 
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/airmass_xx.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/airmass_False.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/tvis_1.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/nside_xx.yml"), TypeError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/nside_neg.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/nside_123.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/nside_absent.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/mask_absent.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/mask_xx.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/weight_xx.yml"), ValueError),
        (os.path.join(os.path.dirname(__file__), "../data/test_ObsCond/nVisYr_tot_xx.yml"), TypeError),
    ],
)
def test_ObsCondition_bad_config(configs, error):
    """Test bad parameters given in config file that should raise Value and Type errors."""
    with pytest.raises(error):
        ObsCondition.make_stage(obs_config_file = configs)









