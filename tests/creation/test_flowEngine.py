"""Tests for FlowEngine."""
import numpy as np
import pytest
import tables_io
from pzflow import Flow
from pzflow.examples import get_example_flow, get_galaxy_data
from rail.creation.engines.flowEngine import FlowCreator, FlowModeler, FlowPosterior
from rail.core.data import TableHandle
from rail.core.stage import RailStage


@pytest.fixture(scope="module")
def catalog_file(tmp_path_factory):
    """Save a subset of the galaxy catalog from PZFlow."""
    file = tmp_path_factory.mktemp("flowEngine_test_files") / "catalog.pq"
    catalog = get_galaxy_data().iloc[:10]
    tables_io.write(catalog, str(file.with_suffix("")), file.suffix[1:])
    return str(file)


@pytest.fixture(scope="module")
def flow_file(tmp_path_factory):
    """Save the example flow from PZFlow."""
    file = tmp_path_factory.mktemp("flowEngine_test_files") / "flow.pzflow.pkl"
    flow = get_example_flow()
    flow.save(file)
    return str(file)


def test_FlowModeler_mags(catalog_file, tmp_path):
    """Test that training a PZFlow Flow doesn't throw any errors.

    Don't calculate colors from the magnitudes.
    """
    # set path for the trained flow
    trained_flow_path = tmp_path / "trained_flow.pzflow.pkl"

    # set the flow parameters
    flow_modeler_params = {
        "name": "flow_modeler_mags",
        "input": catalog_file,
        "model": trained_flow_path,
        "seed": 0,
        "phys_cols": {"redshift": [0, 3]},
        "phot_cols": {
            "g": [16, 32],
            "r": [15, 30],
            "i": [15, 30],
        },
        "calc_colors": {},
        "aliases": {
            "input": "flowModeler_mags_input",
            "model": "flowModeler_mags_model",
        },
    }

    # create the stage to train the flow
    flow_modeler = FlowModeler.make_stage(**flow_modeler_params)

    # train the flow
    flow_modeler.fit_model()

    # load the flow
    trained_flow = Flow(file=trained_flow_path)


def test_FlowModeler_colors(catalog_file, tmp_path):
    """Test that training a PZFlow Flow doesn't throw any errors.

    Calculate colors from the magnitudes.
    """
    # set path for the trained flow
    trained_flow_path = tmp_path / "trained_flow.pzflow.pkl"

    # set the flow parameters
    flow_modeler_params = {
        "name": "flow_modeler_colors",
        "input": catalog_file,
        "model": trained_flow_path,
        "seed": 0,
        "phys_cols": {"redshift": [0, 3]},
        "phot_cols": {
            "g": [16, 32],
            "r": [15, 30],
            "i": [15, 30],
        },
        "calc_colors": {"ref_column_name": "i"},
        "aliases": {
            "input": "flowModeler_colors_input",
            "model": "flowModeler_colors_model",
        },
    }

    # create the stage to train the flow
    flow_modeler = FlowModeler.make_stage(**flow_modeler_params)

    # train the flow
    flow_modeler.fit_model()

    # load the flow
    trained_flow = Flow(file=trained_flow_path)


def test_FlowCreator(flow_file, tmp_path):
    """Test that flow samples and FlowCreator samples are identical.

    We want to make sure the n_samples and seed parameters fully specify the sample.
    We also want to make sure it works if you construct FlowCreator from a flow in
    memory and a flow that is saved to disk.
    """
    # set parameters
    n_samples = 10
    seed = 0

    # load the example flow
    flow = Flow(file=flow_file)

    # draw samples directly from PZFlow
    flow_samples = flow.sample(n_samples, seed=seed)

    # now make a FlowCreator from the flow and draw samples from it
    flowCreator1 = FlowCreator.make_stage(
        name="flowCreator1",
        model=flow,
        output=tmp_path / "samples1.pq",
        n_samples=n_samples,
        aliases={"model": "flowCreator1_model", "output": "flowCreator1_output"},
    )
    flowCreator1_samples = flowCreator1.sample(n_samples, seed=seed).data

    # we will also load the flow from the file via the FlowCreator class
    # and then draw samples from it
    flowCreator2 = FlowCreator.make_stage(
        name="flowCreator2",
        model=flow_file,
        output=tmp_path / "samples2.pq",
        n_samples=n_samples,
        aliases={"model": "flowCreator2_model", "output": "flowCreator2_output"},
    )
    flowCreator2_samples = flowCreator2.sample(n_samples, seed=seed).data

    # check that all samples are the same
    assert np.allclose(flow_samples, flowCreator1_samples)
    assert np.allclose(flow_samples, flowCreator2_samples)
    assert np.allclose(flowCreator1_samples, flowCreator2_samples)


def test_FlowPosterior(catalog_file, flow_file, tmp_path):
    """Test that flow posteriors and FlowPosterior posteriors are identical.

    We want to make sure that FlowPosteriors created from flows in memory and flows
    saved to disk both give the same results as the original flow.
    """
    # load the catalog data
    catalog = tables_io.read(catalog_file)

    # and the example flow
    flow = Flow(file=flow_file)

    # set up a redshift grid to calculate posteriors
    # we will make it small so the test is quick
    grid = np.arange(0, 2.5, 0.5)

    # calculate posteriors directly with PZFlow
    flow_posteriors = flow.posterior(catalog, column="redshift", grid=grid)

    # now make a FlowPosterior object from the flow and calculate posteriors
    flowPosterior1 = FlowPosterior.make_stage(
        name="flowPosterior1",
        model=flow,
        output=tmp_path / "posteriors1.hdf5",
        column="redshift",
        grid=grid,
        marg_rules={"flag": np.nan, "u": lambda row: np.linspace(25, 31, 10)},
        aliases={
            "model": "flowPosterior1_model",
            "input": "flowPosterior1_input",
            "output": "flowPosterior1_output",
        },
    )
    flowPosterior1_posteriors = flowPosterior1.get_posterior(catalog).data
    # pull the posterior values out of qp!
    flowPosterior1_posteriors = flowPosterior1_posteriors.objdata()["yvals"]

    # we will also save the flow and create a FlowPosterior from the saved file
    # then use it to calculate posteriors
    flowPosterior2 = FlowPosterior.make_stage(
        name="flowPosterior2",
        model=flow_file,
        output=tmp_path / "posteriors2.hdf5",
        column="redshift",
        grid=grid,
        marg_rules={"flag": np.nan, "u": lambda row: np.linspace(25, 31, 10)},
        aliases={
            "model": "flowPosterior2_model",
            "input": "flowPosterior2_input",
            "output": "flowPosterior1_output",
        },
    )
    flowPosterior2_posteriors = flowPosterior2.get_posterior(catalog).data
    # pull the posterior values out of qp!
    flowPosterior2_posteriors = flowPosterior2_posteriors.objdata()["yvals"]

    # check that all posteriors are the same
    assert np.allclose(flow_posteriors, flowPosterior1_posteriors)
    assert np.allclose(flow_posteriors, flowPosterior2_posteriors)
    assert np.allclose(flowPosterior1_posteriors, flowPosterior2_posteriors)
