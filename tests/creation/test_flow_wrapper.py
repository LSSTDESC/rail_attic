import numpy as np
from pzflow.examples import example_flow, galaxy_data
from rail.creation import FlowGenerator


def test_flowgenerator_sample():
    flow = example_flow()
    flowGen = FlowGenerator(flow)
    assert np.allclose(
        flow.sample(10, seed=0).values, flowGen.sample(10, seed=0).values
    )


def test_flowgenerator_pz_estimate():
    flow = example_flow()
    flowGen = FlowGenerator(flow)
    data = galaxy_data()[:10]
    grid = np.arange(0, 2.5, 0.5)
    assert np.allclose(
        flow.posterior(data, column="redshift", grid=grid),
        flowGen.pz_estimate(data, grid=grid),
    )
