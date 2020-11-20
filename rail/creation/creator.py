class Creator():
    """
    An object that supplies mock data for redshift estimation experiments.
    The mock data is drawn from a probability distribution defined by the generator,
    with an optional selection function applied.

    generator - an object defining a redshift probability distribution. Must have
                a sample method.
    selection_fn - a selection function to apply to the generated sample
    params - additional information desired to be stored with the instance as a dictionary
    """

    def __init__(self, generator, selection_fn=None, params=None):
        self.generator = generator
        self.selection_fn = selection_fn
        self.params = params

    def sample(self, n_samples, seed=None):
        sample = self.generator.sample(n_samples, seed=seed)
        sample = self.selection_fn(sample) if self.selection_fn is not None else sample
        return sample

