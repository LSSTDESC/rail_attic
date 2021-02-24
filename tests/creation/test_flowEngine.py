import numpy as np
from pzflow.examples import example_flow, galaxy_data
from rail.creation import FlowEngine


def test_flowengine_sample():
    flow = example_flow()
    flowEng = FlowEngine(flow)
    assert np.allclose(
        flow.sample(10, seed=0).values, flowEng.sample(10, seed=0).values
    )


def test_flowengine_pz_estimate():
    flow = example_flow()
    flowEng = FlowEngine(flow)
    data = galaxy_data()[:10]
    grid = np.arange(0, 2.5, 0.5)
    assert np.allclose(
        flow.posterior(data, column="redshift", grid=grid),
        flowEng.get_posterior(data, column="redshift", grid=grid),
    )
