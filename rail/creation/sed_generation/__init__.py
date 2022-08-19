import os
if "SPS_HOME" not in os.environ:
    os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps"
from .generator import *
from .sed_generator import *