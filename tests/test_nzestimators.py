import numpy as np
import os
import copy
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.estimation.algos import NZDir


testszdata = 'tests/data/training_100gal.hdf5'
testphotdata = 'tests/data/validation_10gal.hdf5'
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(key, summarizer_class, summary_kwargs):
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    spec_data = DS.read_file('spec_data', TableHandle, testszdata)
    phot_data = DS.read_file('phot_data', TableHandle, testphotdata)
    summarizer = summarizer_class.make_stage(name=key, **summary_kwargs)
    summary_ens = summarizer.summarize(phot_data, spec_data)                  
    os.remove(summarizer.get_output(summarizer.get_aliased_tag('output'), final_name=True))
    return summary_ens


def test_NZDir():
    summary_config_dict = {}
    summarizer_class = NZDir.NZDir
    results = one_algo("NZDir", summarizer_class, summary_config_dict)
