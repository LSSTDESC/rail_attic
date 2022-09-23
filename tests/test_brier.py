import pytest
import numpy as np
from rail.evaluation.metrics.brier import Brier


def test_brier_base():
    pred = [[1,2,3], [1,2,3]]
    truth = [[1,2,3], [1,2,3]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = np.array([0,0])
    assert isinstance(result, np.ndarray)
    np.testing.assert_equal(result, expected)

def test_brier_1d():
    pred = [[1,2,3]]
    truth = [[1,2,3]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = np.array([0])
    np.testing.assert_equal(result, expected)

def test_brier_input_arrays_different_sizes():
    pred = [[1,2,3], [1,2,3]]
    truth = [[1,2,3], [1,2,3], [1,2,3]]
    brier_obj = Brier(pred, truth)
    with pytest.raises(ValueError):
        result = brier_obj.evaluate()

def test_brier_with_garbage_input():
    pred = [[1,2,3], [1,2,3]]
    truth = "hello world"
    brier_obj = Brier(pred, truth)
    with pytest.raises(ValueError):
        result = brier_obj.evaluate()