import pytest
import os
import filecmp
import numpy as np

from rail.estimation.algos.include_delightPZ.makeConfigParam import makeConfigParam

def test_delight_default_file_gen():
    fakepath = "./"
    config = makeConfigParam(fakepath,None)
    test_file = "test_delight_config.txt"
    with open(test_file, "w") as f:
        f.write(config)
    assert filecmp.cmp(test_file, "./data/delight_default_param.txt")
    os.remove(test_file)
