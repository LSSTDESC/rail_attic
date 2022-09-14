"""Modules to create synthetic data"""
import os
if "SPS_HOME" not in os.environ: # pragma: no cover
    os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps" # pragma: no cover
from rail.creation.degradation import *
from rail.creation.engines import *
from rail.creation.sed_generation import *