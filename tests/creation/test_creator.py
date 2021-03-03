import pytest
import numpy as np
import pandas as pd

from rail.creation import Creator, engines


class sampleEngine(engines.Engine):
    "Create a test engine that is just a normal distribution."

    def sample(self, n_samples, seed=None):
        "Use numpy Random Generator to create reproducible sampler for tests."

        rng = np.random.default_rng(seed)

        return pd.DataFrame(
            rng.normal(size=(n_samples, 7)),
            columns=["redshift", "u", "g", "r", "i", "z", "y"],
        )

    def get_posterior(self, data, column, grid):
        "Return defined set of posteriors for testing"

        output_array = np.ones((len(data), len(grid)))
        test_posterior = output_array * (1.0 / len(grid))

        return test_posterior


@pytest.fixture
def sampleCreator():
    # Instantiate Creator
    sample_eng = sampleEngine()
    test_creator_obj = Creator(sample_eng)
    return test_creator_obj


def test_sample_with_all_options_false(sampleCreator):
    "Test that sample works with seed to create reproducible results"

    # Sample from Creator object
    creator_sample = sampleCreator.sample(1000, seed=42)

    # Create copy of sampler from sampleGenerator
    verify_sampler = np.random.default_rng(seed=42)
    verify_sampler_df = pd.DataFrame(verify_sampler.normal(size=(1000, 7)))

    # Verify samples with same seed and size are the same
    assert np.allclose(creator_sample.values, verify_sampler_df.values)


def test_sample_with_pdf_true(sampleCreator):
    "Test that sample returns pdfs correctly when wanted"

    pdf_sample = sampleCreator.sample(1000, seed=42, include_pdf=True)
    assert np.allclose(
        np.stack(pdf_sample["pz_pdf"].values), np.ones((1000, 101)) * 0.00990099
    )

    grid = np.arange(0, 2.5, 0.5)
    pdf_sample = sampleCreator.sample(1000, seed=42, include_pdf=True, pz_grid=grid)
    assert np.allclose(np.stack(pdf_sample["pz_pdf"].values), np.ones((1000, 5)) * 0.2)


def test_sample_with_selection_fn(sampleCreator):
    "Test that sample is correct after applying selection function."

    rng = np.random.default_rng(42)

    def selection_fn(data, seed=None):
        "Test selection function that just cuts on redshift"

        return data.query("redshift < 2.")

    sampleCreator.degrader = selection_fn

    creator_sample = sampleCreator.sample(1000, seed=42)

    # Create copy of sampler from sampleGenerator
    verify_sampler = np.random.default_rng(seed=42)
    verify_sampler_df = pd.DataFrame(verify_sampler.normal(size=(1000, 7)))
    verify_sampler_df.columns = ["redshift", "u", "g", "r", "i", "z", "y"]
    verify_sampler_keep = verify_sampler_df.query("redshift < 2.")

    verify_sampler_add = np.random.default_rng(seed=rng.integers(1e18))
    verify_sampler_add_df = pd.DataFrame(
        verify_sampler_add.normal(size=(100, 7)),
        columns=["redshift", "u", "g", "r", "i", "z", "y"],
    )
    verify_sample_add_keep = verify_sampler_add_df.query("redshift < 2.")

    verify_concat = pd.concat([verify_sampler_keep, verify_sample_add_keep])
    verify_final = verify_concat.iloc[:1000]

    assert np.allclose(creator_sample.values, verify_final.values)


def test_get_posterior(sampleCreator):

    samples = sampleCreator.sample(1000, seed=42)

    grid = np.arange(20, 30, 0.5)
    pdfs = sampleCreator.get_posterior(samples, column="g", grid=grid)
    assert pdfs.shape == (samples.shape[0], grid.size)
