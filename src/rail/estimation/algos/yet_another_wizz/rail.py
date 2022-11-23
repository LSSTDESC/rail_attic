from __future__ import annotations

import numpy as np

from pandas import DataFrame

from rail.core.data import PqHandle
from rail.core.stage import RailStage


class UniformRandomGenerator(RailStage):
    pass


class SpatialRegionCreator(RailStage):
    pass


class SpatialRegionAssign(RailStage):
    pass


class Counts(RailStage):

    def __init__(self, args, comm=None):
        super().__init__(self, args, comm=comm)
        # construct radial bins
        self.r_min = ...
        self.r_max = ...
        r_steps = ...
        self.r_bins = np.logspace(self.r_min, self.r_max, r_steps + 1)


class CrossCorrelationCounts(Counts):
    """
    TODO.
    Don't forget to define a fixed default cosmology!
    """
    name = "CrossCorrelationCounts"
    config_options = RailStage.config_options.copy()
    inputs = [
        ("reference", PqHandle),
        ("unknown", PqHandle),
        ("randoms", PqHandle)]
    outputs = [("counts", PqHandle)]

    def run(self):
        reference = self.get_data("reference", allow_missing=False)
        unknown = self.get_data("unknown", allow_missing=False)
        randoms = self.get_data("randoms", allow_missing=False)

        self.add_data("counts", ...)

    def __call__(
        self,
        reference: DataFrame,
        unknown: DataFrame,
        randoms: DataFrame
    ) -> DataFrame:
        """
        TODO.
        """
        reference = self.set_data("reference", reference)
        unknown = self.set_data("unknown", unknown)
        randoms = self.set_data("randoms", randoms)
        self.run()
        return self.get_handle("counts")


class AutoCorrelationCounts(Counts):
    """
    TODO.
    Don't forget to define a fixed default cosmology!
    """
    name = "AutoCorrelationCounts"
    config_options = RailStage.config_options.copy()
    inputs = [
        ("data", PqHandle),
        ("randoms", PqHandle)]
    outputs = [("counts", PqHandle)]

    def __init__(self, args, comm=None):
        super().__init__(self, args, comm=comm)

    def run(self):
        data = self.get_data("data", allow_missing=False)
        randoms = self.get_data("randoms", allow_missing=False)

        self.add_data("counts", ...)

    def __call__(
        self,
        data: DataFrame,
        randoms: DataFrame
    ) -> DataFrame:
        """
        TODO.
        """
        data = self.set_data("data", data)
        randoms = self.set_data("randoms", randoms)
        self.run()
        return self.get_handle("counts")


class CorrelationCountsMerger(RailStage):
    pass


class CorrelationMesuarements(RailStage):
    pass


class YAWInterface(RailStage):
    pass


# TODO: don't forget entry in README if that is still maintained.
