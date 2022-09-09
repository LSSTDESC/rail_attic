"""RailStages that degrade synthetic samples of photometric data"""

from rail.creation.degradation.grid_selection import GridSelection
from rail.creation.degradation.lsst_error_model import LSSTErrorModel
from rail.creation.degradation.quantityCut import QuantityCut
from rail.creation.degradation.spectroscopic_degraders import (
    InvRedshiftIncompleteness,
    LineConfusion,
)
from rail.creation.degradation.spectroscopic_selections import (
    SpecSelection,
    SpecSelection_BOSS,
    SpecSelection_DEEP2,
    SpecSelection_GAMA,
    SpecSelection_HSC,
    SpecSelection_VVDSf02,
    SpecSelection_zCOSMOS,
)
