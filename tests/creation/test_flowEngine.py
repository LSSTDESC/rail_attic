import numpy as np
from pzflow.examples import get_example_flow, get_galaxy_data
from rail.creation.engines import FlowEngine


def test_flowengine_sample():
    """Test that flow samples and flowEng samples are the same."""

    n_samples = 10
    seed = 0

    flow = get_example_flow()
    flow_samples = flow.sample(n_samples, seed=seed).to_numpy()

    flowEng = FlowEngine(flow)
    flowEng_samples = flowEng.sample(n_samples, seed=seed).to_numpy()

    assert np.allclose(flow_samples, flowEng_samples)


def test_flowengine_pz_estimate():
    """Test that flow posteriors and flowEng posteriors are the same."""

    data = get_galaxy_data().iloc[:10, :]
    grid = np.arange(0, 2.5, 0.5)

    flow = get_example_flow()
    flow_pdfs = flow.posterior(data, column="redshift", grid=grid)

    flowEng = FlowEngine(flow)
    flowEng_pdfs = flowEng.get_posterior(data, column="redshift", grid=grid)
    flowEng_pdfs = flowEng_pdfs.objdata()["yvals"]

    assert np.allclose(flow_pdfs, flowEng_pdfs)
