"""RailStages that degrade synthetic samples of photometric data"""

from rail.creation.degradation.grid_selection import GridSelection
from rail.creation.degradation.lsst_error_model import LSSTErrorModel
from rail.creation.degradation.quantityCut import QuantityCut
from rail.creation.degradation.spectroscopic_degraders import (
    InvRedshiftIncompleteness,
    LineConfusion,
)
