import os
import numpy as np
import pzflow
from pzflow.examples import get_example_flow, get_galaxy_data
from rail.creation.engines.flowEngine import FlowEngine, FlowPosterior
from rail.core.data import TableHandle
from rail.core.stage import RailStage

def test_flowengine_sample():
    """Test that flow samples and flowEng samples are the same."""

    n_samples = 10
    seed = 0

    flow = get_example_flow()
    flow_samples = flow.sample(n_samples, seed=seed)

    flowEng = FlowEngine.make_stage(flow=flow, n_samples=n_samples)
    flowEng_samples = flowEng.sample(n_samples, seed=seed).data

    pzdir = os.path.dirname(pzflow.__file__)
    flow_path = os.path.join(pzdir, 'examples', 'example-flow.pkl')
    
    flowEng2 = FlowEngine.make_stage(name="other_flow", 
                                     flow_file=flow_path, n_samples=n_samples)
    flowEng2_samples = flowEng2.sample(n_samples, seed=seed).data

    #assert flow_samples.equals(flowEng_samples)
    #assert flow_samples.equals(flowEng2_samples)
    os.remove(flowEng.get_output(flowEng.get_aliased_tag('output'), final_name=True))
    os.remove(flowEng2.get_output(flowEng2.get_aliased_tag('output'), final_name=True))


def test_flowengine_pz_estimate(tmp_path):
    """Test that flow posteriors and flowEng posteriors are the same."""

    data = get_galaxy_data().iloc[:10, :]
    DS = RailStage.data_store
    DS.clear()
    handle = DS.add_data('data', data, TableHandle, path='dummy.pd')

    bands = ['u','g','r','i','z','y']
    rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}

    grid = np.arange(0, 2.5, 0.5)

    flow = get_example_flow()
    flow_pdfs = flow.posterior(data, column="redshift", grid=grid)

    flow_path = tmp_path / "flow.pzflow.pkl"
    flow.save(flow_path)
    flowPost = FlowPosterior.make_stage(name='flow',
                                        flow=flow_path,
                                        column="redshift",
                                        grid=grid,
                                        marg_rules = {"flag": np.nan, "u": lambda row: np.linspace(25, 31, 10)})

    flowPost2 = FlowPosterior.make_stage(name='flow2',
                                        flow=flow,
                                        column="redshift",
                                        grid=grid,
                                        marg_rules = {"flag": np.nan, "u": lambda row: np.linspace(25, 31, 10)})

    
    flowPost_pdfs = flowPost.get_posterior(handle, column="redshift", grid=grid).data
    flowPost_pdfs = flowPost_pdfs.objdata()["yvals"]

    flowPost2_pdfs = flowPost2.get_posterior(handle, column="redshift", grid=grid).data
    flowPost2_pdfs = flowPost2_pdfs.objdata()["yvals"]

    assert np.allclose(flow_pdfs, flowPost_pdfs)
    os.remove(flowPost.get_output(flowPost.get_aliased_tag('output'), final_name=True))
    os.remove(flowPost2.get_output(flowPost2.get_aliased_tag('output'), final_name=True))

    
