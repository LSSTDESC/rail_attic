import numpy as np
import pandas as pd
import pytest
import qp
from rail.creation import Creator, Engine


class SampleEngine(Engine):
    """Create a test engine that is just a normal distribution."""

    def sample(self, n_samples, seed=None):
        """Use numpy Random Generator to create reproducible sampler for tests."""

        rng = np.random.default_rng(seed)
        sample = rng.normal(size=(n_samples, 7))

        return pd.DataFrame(
            sample,
            columns=["redshift"] + list("ugrizy"),
        )

    def get_posterior(self, data, column, grid):
        """Return defined set of posteriors for testing"""

        output_array = np.ones((len(data), len(grid)))
        test_posterior = output_array * (1.0 / len(grid))

        return qp.Ensemble(qp.interp, data={"xvals": grid, "yvals": test_posterior})


@pytest.fixture
def sampleCreator():
    # Instantiate Creator
    return Creator(SampleEngine())


def test_sample_with_all_options_false(sampleCreator):
    """Test that sample works with seed to create reproducible results"""

    n_samples = 1000
    seed = 42

    # Sample from Creator object
    creator_sample = sampleCreator.sample(n_samples, seed=seed)

    # Create copy of sampler from sampleGenerator
    verify_sampler = np.random.default_rng(seed=seed)
    verify_df = pd.DataFrame(verify_sampler.normal(size=(n_samples, 7)))

    # Verify samples with same seed and size are the same
    assert np.allclose(creator_sample.values, verify_df.values)


def test_sample_with_selection_fn(sampleCreator):
    """Test that sample is correct after applying selection function."""

    def selection_fn(data, seed=None):
        """Test selection function that just cuts on redshift"""

        return data.query("redshift < 2.")

    sampleCreator.degrader = selection_fn

    # get the sample from the Creator
    sample = sampleCreator.sample(1000, seed=42)

    # check correct shape
    assert sample.shape == (1000, 7)

    # check that cut was applied
    assert sample["redshift"].max() < 2


def test_posterior_shapes(sampleCreator):
    """Test posterior shapes"""

    samples = sampleCreator.sample(1000, seed=42)

    grid = np.arange(20, 30, 0.5)
    pdfs = sampleCreator.get_posterior(samples, column="g", grid=grid)

    assert pdfs.shape == (samples.shape[0],)
    assert pdfs.objdata()["yvals"].shape == (samples.shape[0], grid.size)
