import logging
import numpy as np
from deprecated import deprecated
from .base import MetricEvaluator

@deprecated(
    reason="""
    This implementation of the Brier metric is deprecated.
    Please use qp.metrics.calculate_brier(prediction, truth) from the qp-prob package.
    """,
    category=DeprecationWarning)
class Brier(MetricEvaluator):
    """ Brier score """
    def __init__(self, prediction, truth):
        """
        Parameters
        ----------
        prediction: NxM array, float
            Predicted probability for N celestial objects to have a redshift in
            one of M bins. The sum of values along each row N should be 1.
        truth: NxM array, int
            True redshift values for N celestial objects, where Mth bin for the
            true redshift will have value 1, all other bins will have a value of
            0.

        Notes
        -----
        Based on https://en.wikipedia.org/wiki/Brier_score#Original_definition_by_Brier
        """
        super().__init__(None)

        self._prediction = prediction
        self._truth = truth
        self._axis_for_summation = None #axis to sum for metric calculation

    def evaluate(self):
        """
        Evaluate the Brier score for N celestial objects.

        Returns
        -------
        Brier metric: float in the interval [0,2]
            The result of calculating the Brier metric.
        """

        self._manipulate_data()
        self._validate_data()
        return self._calculate_metric()

    def _manipulate_data(self):
        """
        Placeholder for data manipulation as required. i.e. converting from
        qp.ensemble objects into np.array objects.
        """

        # Attempt to convert the input variables into np.arrays
        self._prediction = np.array(self._prediction)
        self._truth = np.array(self._truth)

    def _validate_data(self):
        """
        Strictly for data validation - no calculations or data structure
        changes.

        Raises
        ------
        TypeError if either prediction or truth input could not be converted
        into a numeric Numpy array

        ValueError if the prediction and truth arrays do not have the same
        numpy.shape.

        Warning
        -------
        Logs a warning message if the input predictions do not each sum to 1.
        """

        # Raise TypeError exceptions if the inputs were not translated to
        # numeric np.arrays
        if not np.issubdtype(self._prediction.dtype, np.number):
            raise TypeError("Input prediction array could not be converted to a Numpy array")
        if not np.issubdtype(self._truth.dtype, np.number):
            raise TypeError("Input truth array could not be converted to a Numpy array")

        # Raise ValueError if the arrays have different shapes
        if self._prediction.shape != self._truth.shape:
            raise ValueError("Input prediction and truth arrays do not have the same shape")

        # Log a warning if the N rows of the input prediction do not each sum to
        # 1. Note: For 1d arrays, a sum along axis = 1 will fail, so we set
        # self._axis_for_summation appropriately for that case
        self._axis_for_summation = 0 if self._prediction.ndim == 1 else 1
        if not np.allclose(np.sum(self._prediction, axis=self._axis_for_summation), 1.):
            logging.warning("Input predictions do not sum to 1.")

    def _calculate_metric(self):
        """
        Calculate the Brier metric for the input data.
        """
        return np.mean(np.sum((self._prediction - self._truth)**2, axis=self._axis_for_summation))
