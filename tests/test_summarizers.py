import numpy as np
import os
import copy
import pytest
from rail.core.stage import RailStage
from rail.core.data import QPHandle
from rail.summarization.algos import naiveStack, pointEstimateHist

testdata = 'tests/data/output_BPZ_lite.fits'
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(key, summarizer_class, summary_kwargs):
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    test_data = DS.read_file('test_data', QPHandle, testdata)
    summarizer = summarizer_class.make_stage(name=key, **summary_kwargs)    
    summary_ens = summarizer.summarize(test_data)                  
    return summary_ens


def test_naive_stack():
    summary_config_dict = {}
    summarizer_class = naiveStack.NaiveStack
    results = one_algo("NaiveStack", summarizer_class, summary_config_dict)


def test_point_estimate_hist():
    summary_config_dict = {}
    summarizer_class = pointEstimateHist.PointEstimateHist
    results = one_algo("PointEstimateHist", summarizer_class, summary_config_dict)
