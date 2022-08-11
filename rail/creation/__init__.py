"""Modules to create synthetic data"""
import os
os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps"
from .degradation import *
from .engines import *
from .sed_generation import *