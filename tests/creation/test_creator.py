import pytest
import numpy as np
import pandas as pd
from rail.creation import Creator


class sampleGenerator(object):
    "Create a test generator that is just a normal distribution."

    def sample(self, n_samples, seed=None):
        "Use RandomState to create reproducible sampler for testing."

        random_state = np.random.RandomState(seed=seed)

        return pd.DataFrame(random_state.normal(size=(n_samples, 7)))

    def pz_estimate(self, data, zmin, zmax, dz, convolve_err=False):
        "Return defined set of posteriors for testing"

        z_steps = np.arange(zmin, zmax+dz, dz)
        output_array = np.ones((len(data), len(z_steps)))
        test_posterior = output_array*(1./len(z_steps))

        if convolve_err is False:
            return test_posterior


@pytest.fixture
def generator():
    # Instantiate Creator
    sample_gen = sampleGenerator()
    test_creator_obj = Creator(sample_gen)
    return test_creator_obj


def test_sample_with_all_options_false(generator):
    "Test that sample works with seed to create reproducible results"

    # Sample from Creator object
    creator_sample = generator.sample(1000, seed=42)

    # Create copy of sampler from sampleGenerator
    verify_sampler = np.random.RandomState(seed=42)
    verify_sampler_df = pd.DataFrame(verify_sampler.normal(size=(1000, 7)))

    # Verify samples with same seed and size are the same
    np.testing.assert_array_equal(creator_sample, verify_sampler_df)


def test_sample_with_pdf_true(generator):
    "Test that sample returns pdfs correctly when wanted"
    pdf_sample = generator.sample(1000, seed=42, include_pdf=True,
                                  zmin=0, zmax=2, dz=0.5)
    np.testing.assert_array_equal(np.stack(pdf_sample['pz_pdf'].values),
                                  np.ones((1000, 5))*0.2)
