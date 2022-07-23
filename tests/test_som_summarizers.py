import os
from rail.core.stage import RailStage
from rail.core.data import TableHandle
from rail.estimation.algos import simpleSOM


testszdata = 'tests/data/training_100gal.hdf5'
testphotdata = 'tests/data/validation_10gal.hdf5'
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(key, inform_class, summarizer_class, summary_kwargs):
    """
    A basic test of running an summaizer subclass
    Run summarize
    """
    spec_data = DS.read_file('spec_data', TableHandle, testszdata)
    phot_data = DS.read_file('phot_data', TableHandle, testphotdata)
    informer = inform_class.make_stage(name=f"inform_{key}", model="tmpsom.pkl")
    informer.inform(spec_data)
    summarizerr = summarizer_class.make_stage(name=key, model=informer.get_handle('model'), **summary_kwargs)
    summary_ens = summarizerr.summarize(phot_data, spec_data)
    os.remove(summarizerr.get_output(summarizerr.get_aliased_tag('output'), final_name=True))
    # test loading model by name rather than via handle
    summarizer2 = summarizer_class.make_stage(name=key, model='tmpsom.pkl', **summary_kwargs)
    summ_en = summarizer2.summarize(phot_data, spec_data)
    os.remove(summarizer2.get_output(summarizer2.get_aliased_tag('output'), final_name=True))
    os.remove("tmpsom.pkl")
    return summary_ens


def test_SimpleSOM():
    summary_config_dict = {}
    inform_class = simpleSOM.Inform_SimpleSOMSummarizer
    summarizerclass = simpleSOM.SimpleSOMSummarizer
    _ = one_algo("SimpleSOM", inform_class, summarizerclass, summary_config_dict)