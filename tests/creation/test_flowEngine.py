import numpy as np
from pzflow.examples import get_example_flow, get_galaxy_data
from rail.creation.engines import FlowEngine


def test_flowengine_sample():
    flow = get_example_flow()
    flowEng = FlowEngine(flow)
    assert np.allclose(
        flow.sample(10, seed=0).values, flowEng.sample(10, seed=0).values
    )


def test_flowengine_pz_estimate():
    flow = get_example_flow()
    flowEng = FlowEngine(flow)
    data = get_galaxy_data()[:10]
    grid = np.arange(0, 2.5, 0.5)
    assert np.allclose(
        flow.posterior(data, column="redshift", grid=grid),
        flowEng.get_posterior(data, column="redshift", grid=grid),
    )
