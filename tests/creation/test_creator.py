import unittest
import numpy as np
from rail.creation import Creator


class sampleGenerator(object):
    "Create a test generator that is just a normal distribution."

    def sample(self, n_samples, seed=None):
        "Use RandomState to create reproducible sampler for testing."

        random_state = np.random.RandomState(seed=seed)

        return random_state.normal(size=n_samples)


class testCreator(unittest.TestCase):

    def test_creator(self):
        "Test that sample works with seed to create reproducible results"

        # Instantiate Creator
        sample_gen = sampleGenerator()
        test_creator_obj = Creator(sample_gen)

        # Sample from Creator object
        creator_sample = test_creator_obj.sample(1000, seed=42)

        # Create copy of sampler from sampleGenerator
        verify_sampler = np.random.RandomState(seed=42)

        # Verify samples with same seed and size are the same
        np.testing.assert_array_equal(creator_sample, verify_sampler.normal(size=1000))

if __name__ == "__main__":

    unittest.main()
