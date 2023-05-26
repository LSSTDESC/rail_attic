import logging

import numpy as np
import pytest

from rail.evaluation.metrics.brier import Brier

LOGGER = logging.getLogger(__name__)


def test_brier_base():
    """
    Test the base case, ensure output is expected.
    """
    pred = [[1, 0, 0], [1, 0, 0]]
    truth = [[1, 0, 0], [0, 1, 0]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = 1.0
    assert np.isclose(result, expected)


def test_brier_result_is_scalar():
    """
    Verify output is scalar for input of NxM.
    """
    pred = [[1, 0, 0], [0, 1, 0], [0, 0.5, 0.5]]
    truth = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    assert np.isscalar(result)


def test_brier_base_with_non_integers():
    """
    Verify output for non-integer prediction values.
    """
    pred = [[0.5, 0.5, 0]]
    truth = [[1, 0, 0]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = 0.5
    assert np.isclose(result, expected)


def test_brier_max_result():
    """
    Base case where prediction is completely wrong, should produce maximum
    possible result value, 2.
    """
    pred = [[0, 1, 0], [1, 0, 0]]
    truth = [[1, 0, 0], [0, 1, 0]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = 2.0
    assert np.isclose(result, expected)


def test_brier_min_result():
    """
    Base case where prediction is perfect, should produce minimum possible
    result value, 0.
    """
    pred = [[1, 0, 0], [0, 1, 0]]
    truth = [[1, 0, 0], [0, 1, 0]]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = 0.0
    assert np.isclose(result, expected)


def test_brier_input_arrays_different_sizes():
    """
    Verify exception is raised when input arrays are different sizes.
    """
    pred = [[1, 0, 0], [0, 1, 0]]
    truth = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    brier_obj = Brier(pred, truth)
    with pytest.raises(ValueError):
        _ = brier_obj.evaluate()


def test_brier_with_garbage_prediction_input():
    """
    Verify exception is raised when prediction input is non-numeric.
    """
    pred = ["foo", "bar"]
    truth = [[1, 0, 0], [0, 1, 0]]
    brier_obj = Brier(pred, truth)
    with pytest.raises(TypeError):
        _ = brier_obj.evaluate()


def test_brier_with_garbage_truth_input():
    """
    Verify exception is raised when truth input is non-numeric.
    """
    pred = [[1, 0, 0], [0, 1, 0]]
    truth = ["hello sky", "goodbye ground"]
    brier_obj = Brier(pred, truth)
    with pytest.raises(TypeError):
        _ = brier_obj.evaluate()


def test_brier_prediction_does_not_sum_to_one(caplog):
    """
    Verify exception is raised when prediction input rows don't sum to 1 This
    also verifies that while the total sum of values in the prediction array sum
    to 2, the individual rows do not, and thus logs a warning
    """
    pred = [[1, 0.0001, 0], [0, 0.9999, 0]]
    truth = [[1, 0, 0], [0, 1, 0]]
    LOGGER.info("Testing now...")
    brier_obj = Brier(pred, truth)
    with caplog.at_level(logging.WARNING):
        _ = brier_obj.evaluate()
    assert "Input predictions do not sum to 1" in caplog.text


def test_brier_1d_prediction_does_not_sum_to_one(caplog):
    """
    Verify exception is raised when 1d prediction input rows don't sum to 1
    """
    pred = [0.3, 0.8, 0]
    truth = [1, 0, 0]
    LOGGER.info("Testing now...")
    brier_obj = Brier(pred, truth)
    with caplog.at_level(logging.WARNING):
        _ = brier_obj.evaluate()
    assert "Input predictions do not sum to 1" in caplog.text


def test_brier_1d():
    """
    Verify 1 dimensional input produced the correct output. This exercises the
    condition in brier._calculate_metric that changes the axis upon which the
    np.mean operates.
    """
    pred = [1, 0, 0]
    truth = [1, 0, 0]
    brier_obj = Brier(pred, truth)
    result = brier_obj.evaluate()
    expected = 0.0
    assert np.isclose(result, expected)
