from rail.creation.generator import Generator


class FlowGenerator(Generator):
    def __init__(self, flow):
        self.flow = flow

    def sample(self, n_samples: int, seed: int = None):
        return self.flow.sample(nsamples=n_samples, seed=seed)

    def pz_estimate(self, data, grid, **kwargs):
        return self.flow.posterior(data, column="redshift", grid=grid)
