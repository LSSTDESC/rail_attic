import numpy as np
from pzflow.examples import get_example_flow, get_galaxy_data
from rail.creation.engines import FlowEngine, FlowPosterior
from rail.core.data import DATA_STORE, TableHandle

def test_flowengine_sample():
    """Test that flow samples and flowEng samples are the same."""

    n_samples = 10
    seed = 0

    flow = get_example_flow()
    flow_samples = flow.sample(n_samples, seed=seed)

    flowEng = FlowEngine.make_stage(flow=flow, n_samples=n_samples)
    flowEng_samples = flowEng.sample(n_samples, seed=seed).data
    
    assert flow_samples.equals(flowEng_samples)


def test_flowengine_pz_estimate():
    """Test that flow posteriors and flowEng posteriors are the same."""

    data = get_galaxy_data().iloc[:10, :]
    DS = DATA_STORE()
    DS.add_data('data', data, TableHandle, path='dummy.pd')

    bands = ['u','g','r','i','z','y']
    rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}

    grid = np.arange(0, 2.5, 0.5)

    flow = get_example_flow()
    flow_pdfs = flow.posterior(data, column="redshift", grid=grid)

    flowPost = FlowPosterior.make_stage(name='flow',
                                        flow=flow,
                                        column="redshift",
                                        grid=grid,
                                        marg_rules = {"flag": np.nan, "u": lambda row: np.linspace(25, 31, 10)})

    flowPost_pdfs = flowPost.get_posterior(DS['data'], column="redshift", grid=grid).data
    flowPost_pdfs = flowPost_pdfs.objdata()["yvals"]

    assert np.allclose(flow_pdfs, flowPost_pdfs)
