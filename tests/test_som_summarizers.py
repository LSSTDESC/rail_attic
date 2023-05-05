import os

import numpy as np
import qp

from rail.core.data import TableHandle
from rail.core.stage import RailStage
from rail.core.utils import RAILDIR
from rail.estimation.algos import simpleSOM

testszdata = os.path.join(RAILDIR, "rail/examples_data/testdata/training_100gal.hdf5")
testphotdata = os.path.join(RAILDIR, "rail/examples_data/testdata/validation_10gal.hdf5")
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(key, inform_class, summarizer_class, summary_kwargs):
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    spec_data = DS.read_file("spec_data", TableHandle, testszdata)
    phot_data = DS.read_file("phot_data", TableHandle, testphotdata)
    informer = inform_class.make_stage(
        name=f"inform_{key}",
        model=f"tmpsom_{key}.pkl",
        aliases={"model": "som_inform_test_model", "input": "som_inform_test_input"},
        **summary_kwargs,
    )
    informer.inform(spec_data)
    summarizerr = summarizer_class.make_stage(
        name=key,
        model=informer.get_handle("model"),
        aliases={
            "model": "som_summarize_test_model",
            "input": "som_summarize_test_input",
            "spec_input": "som_summarize_test_spec_input",
        },
        **summary_kwargs,
    )
    summary_ens = summarizerr.summarize(phot_data, spec_data)
    os.remove(summarizerr.get_output(summarizerr.get_aliased_tag("output"), final_name=True))
    # test loading model by name rather than via handle
    summarizer2 = summarizer_class.make_stage(
        name=key,
        model=f"tmpsom_{key}.pkl",
        aliases={
            "model": "som_summarize_test2_model",
            "input": "som_summarize_test2_input",
            "spec_input": "som_summarize_test2_spec_input",
        },
    )
    _ = summarizer2.summarize(phot_data, spec_data)
    fid_ens = qp.read(summarizer2.get_output(summarizer2.get_aliased_tag("single_NZ"), final_name=True))
    meanz = fid_ens.mean().flatten()
    assert np.isclose(meanz[0], 0.1493592786, atol=0.025)
    os.remove(summarizer2.get_output(summarizer2.get_aliased_tag("output"), final_name=True))
    os.remove(f"tmpsom_{key}.pkl")
    return summary_ens


def test_SimpleSOM():
    summary_config_dict = {"m_dim": 21, "n_dim": 21, "column_usage": "colors"}
    inform_class = simpleSOM.Inform_SimpleSOMSummarizer
    summarizerclass = simpleSOM.SimpleSOMSummarizer
    _ = one_algo("SimpleSOM", inform_class, summarizerclass, summary_config_dict)


def test_SimpeSOM_with_mag_and_colors():
    summary_config_dict = {
        "m_dim": 21,
        "n_dim": 21,
        "column_usage": "magandcolors",
        "objid_name": "id",
    }
    inform_class = simpleSOM.Inform_SimpleSOMSummarizer
    summarizerclass = simpleSOM.SimpleSOMSummarizer
    _ = one_algo("SimpleSOM_wmag", inform_class, summarizerclass, summary_config_dict)
