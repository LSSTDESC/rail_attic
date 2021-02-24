# this is a subclass of Engine that wraps a pzflow Flow so that it can
# be used as the engine of a Creator object.
# If we create more subclasses of Engine, we should move them to a subdirectory
# However, this is currently the only one.

from rail.creation.engine import Engine


class FlowEngine(Engine):
    def __init__(self, flow):
        self.flow = flow

    def sample(self, n_samples: int, seed: int = None):
        return self.flow.sample(nsamples=n_samples, seed=seed)

    def get_posterior(self, data, column, grid, **kwargs):
        return self.flow.posterior(data, column=column, grid=grid)
