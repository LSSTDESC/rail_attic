import copy
import os

import numpy as np
import pytest

from rail.core.data import QPHandle
from rail.core.stage import RailStage
from rail.core.utils import RAILDIR
from rail.estimation.algos import naiveStack, pointEstimateHist, varInference

testdata = os.path.join(RAILDIR, "rail/examples_data/testdata/output_BPZ_lite.fits")
DS = RailStage.data_store


def one_algo(key, summarizer_class, summary_kwargs):
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    DS.__class__.allow_overwrite = True
    test_data = DS.read_file("test_data", QPHandle, testdata)
    summarizer = summarizer_class.make_stage(name=key, **summary_kwargs)
    summary_ens = summarizer.summarize(test_data)
    os.remove(summarizer.get_output(summarizer.get_aliased_tag("output"), final_name=True))
    os.remove(summarizer.get_output(summarizer.get_aliased_tag("single_NZ"), final_name=True))
    return summary_ens


def test_naive_stack():
    summary_config_dict = {}
    summarizer_class = naiveStack.NaiveStack
    results = one_algo("NaiveStack", summarizer_class, summary_config_dict)


def test_point_estimate_hist():
    summary_config_dict = {}
    summarizer_class = pointEstimateHist.PointEstimateHist
    results = one_algo("PointEstimateHist", summarizer_class, summary_config_dict)


def test_var_inference_stack():
    summary_config_dict = {}
    summarizer_class = varInference.VarInferenceStack
    results = one_algo("VariationalInference", summarizer_class, summary_config_dict)
