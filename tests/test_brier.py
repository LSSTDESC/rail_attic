import pytest
import numpy as np
from rail.evaluation.metrics.brier import Brier


def test_brier_base():
    """
    Test the base case, ensure output is expected.
    """
    pred = [[1,0,0], [1,0,1]]
    truth = [[1,0,0], [0,1,0]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = np.array([0,1])
    assert isinstance(result, np.ndarray)
    np.testing.assert_equal(result, expected)

def test_brier_result_has_correct_dimensions():
    """
    Verify the dimensionality of the output. Given input of NxM, expect output
    of Nx1.
    """
    pred = [[1,0,0], [0,1,0], [0,0,1]]
    truth = [[1,0,0], [0,1,0], [0,0,1]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    np.testing.assert_equal(result.shape, (3,))

def test_brier_base_with_non_integers():
    """
    Verify output for non-integer prediction values.
    """
    pred = [[0.5,0.5,0]]
    truth = [[1,0,0]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = np.array([0.16666667])
    assert np.isclose(result, expected)

def test_brier_input_arrays_different_sizes():
    """
    Verify exception is raised when input arrays are different sizes.
    """
    pred = [[1,0,0], [0,1,0]]
    truth = [[1,0,0], [0,1,0], [0,0,0]]
    brier_obj = Brier(pred, truth)
    with pytest.raises(ValueError):
        _ = brier_obj.evaluate()

def test_brier_with_garbage_prediction_input():
    """
    Verify exception is raised when prediction input is non-numeric.
    """
    pred = ["foo", "bar"]
    truth = [[1,0,0],[0,1,0]]
    brier_obj = Brier(pred, truth)
    with pytest.raises(TypeError):
        _ = brier_obj.evaluate()

def test_brier_with_garbage_truth_input():
    """
    Verify exception is raised when truth input is non-numeric.
    """
    pred = [[1,0,0], [0,1,0]]
    truth = ["hello sky", "goodbye ground"]
    brier_obj = Brier(pred, truth)
    with pytest.raises(TypeError):
        _ = brier_obj.evaluate()

def test_brier_1d():
    """
    Verify 1 dimensional input produced the correct output. This exercises the
    condition in brier._calculate_metric that changes the axis upon which the
    np.mean operates.
    """
    pred = [1,0,0]
    truth = [1,0,0]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = np.array([0])
    np.testing.assert_equal(result, expected)

def test_brier_1d_result_has_correct_shape():
    """
    Verify 1 dimensional input produces a 1 dimensional output.
    """
    pred = [1,0,0]
    truth = [1,0,0]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    np.testing.assert_equal(result.shape, (1,))
