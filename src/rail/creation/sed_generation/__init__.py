import os
if "SPS_HOME" not in os.environ: # pragma: no cover
    os.environ["SPS_HOME"] = "/opt/hostedtoolcache/Python/fsps" # pragma: no cover
from .generator import *
from .sed_generator import *