"""Modules to create synthetic data"""
import os
if "SPS_HOME" not in os.environ:
    os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps"
from .degradation import *
from .engines import *
from .sed_generation import *