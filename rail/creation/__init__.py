"""Modules to create synthetic data"""
import os
if "SPS_HOME" not in os.environ: # pragma: no cover
    os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps" # pragma: no cover
from .degradation import *
from .engines import *
from .sed_generation import *