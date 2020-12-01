import pytest
import numpy as np
import pandas as pd
from rail.creation import Creator


class sampleGenerator(object):
    "Create a test generator that is just a normal distribution."

    def sample(self, n_samples, seed=None):
        "Use RandomState to create reproducible sampler for testing."

        random_state = np.random.RandomState(seed=seed)

        return pd.DataFrame(random_state.normal(size=(n_samples, 7)),
                            columns=['redshift', 'u', 'g', 'r', 'i', 'z', 'y'])

    def pz_estimate(self, data, zmin, zmax, dz, convolve_err=False):
        "Return defined set of posteriors for testing"

        z_steps = np.arange(zmin, zmax+dz, dz)
        output_array = np.ones((len(data), len(z_steps)))
        test_posterior = output_array*(1./len(z_steps))

        if convolve_err is False:
            return test_posterior


@pytest.fixture
def sampleCreator():
    # Instantiate Creator
    sample_gen = sampleGenerator()
    test_creator_obj = Creator(sample_gen)
    return test_creator_obj


def test_sample_with_all_options_false(sampleCreator):
    "Test that sample works with seed to create reproducible results"

    # Sample from Creator object
    creator_sample = sampleCreator.sample(1000, seed=42)

    # Create copy of sampler from sampleGenerator
    verify_sampler = np.random.RandomState(seed=42)
    verify_sampler_df = pd.DataFrame(verify_sampler.normal(size=(1000, 7)))

    # Verify samples with same seed and size are the same
    np.testing.assert_array_equal(creator_sample, verify_sampler_df)


def test_sample_with_pdf_true(sampleCreator):
    "Test that sample returns pdfs correctly when wanted"
    pdf_sample = sampleCreator.sample(1000, seed=42, include_pdf=True,
                                      zmin=0, zmax=2, dz=0.5)
    np.testing.assert_array_equal(np.stack(pdf_sample['pz_pdf'].values),
                                  np.ones((1000, 5))*0.2)


def test_params_dict_length_assert():
    """
    Test params assignment raises error
    if bands length is not 2x err_params length
    """

    with pytest.raises(AssertionError):
        test_params = {}
        test_params['bands'] = ['a', 'b', 'c']
        # Set up creator
        Creator(sampleGenerator(), params=test_params)


def test_params_dict_assignment():
    "Test that we can alter err_params but leave bands alone"

    test_params = {}
    test_err_params = {'gamma_u': 0.05, 'gamma_g': 0.05,
                       'gamma_r': 0.05, 'gamma_i': 0.05,
                       'gamma_z': 0.05, 'gamma_y': 0.05,
                       'm5_u': 29., 'm5_g': 29.,
                       'm5_r': 29., 'm5_i': 29.,
                       'm5_z': 29., 'm5_y': 29.}
    test_params['err_params'] = test_err_params

    test_creator = Creator(sampleGenerator(), params=test_params)

    test_params['bands'] = ['u', 'g', 'r', 'i', 'z', 'y']

    assert test_params == test_creator.params


def test_include_err(sampleCreator):

    # Sample from Creator object
    creator_sample = sampleCreator.sample(1000, seed=42,
                                          include_err=True, err_seed=17)

    # Create copy of sampler from sampleGenerator
    verify_sampler = sampleCreator.sample(1000, seed=42)
    rand_state = np.random.RandomState(seed=17)

    err_params = sampleCreator.params['err_params']
    x_u_test = 10**(0.4*(verify_sampler['u'] - err_params['m5_u']))
    u_err = np.sqrt((0.04*err_params['gamma_u'])*x_u_test +
                    err_params['gamma_u']*x_u_test**2)
    u_new_sample = rand_state.normal(verify_sampler['u'], u_err)

    np.testing.assert_array_almost_equal(u_new_sample,
                                         creator_sample['u'].values)
