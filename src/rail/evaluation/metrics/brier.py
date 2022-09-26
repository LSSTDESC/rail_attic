import numpy as np
from .base import MetricEvaluator

class Brier(MetricEvaluator):
    """ Brier score """
    def __init__(self, prediction, truth):
        """
        Parameters
        ----------
        prediction: NxM array, float
            Predicted probability for N celestial objects to have a redshift in
            one of M bins.
        truth: NxM array, int
            True redshift values for N celestial objects, where Mth bin for the
            true redshift will have value 1, all other bins will have a value of
            0.

        Notes
        -----
        Based on https://en.wikipedia.org/wiki/Brier_score#Definition
        """
        super().__init__(None)

        self._prediction = prediction
        self._truth = truth

    def evaluate(self):
        """
        Evaluate the Brier score for N celestial objects.

        Returns
        -------
        Brier metric: Nx1 numpy.ndarray, float
            The result of calculating the Brier metric.
        """

        self._manipulate_data()
        self._validate_data()
        return self._calculate_metric()

    def _manipulate_data(self):
        """
        Placeholder for data manipulation as required. i.e. converting from
        qp.ensamble objects into np.array objects.
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
        """

        # Raise TypeError exceptions if the inputs were not translated to
        # numeric np.arrays
        if not np.issubdtype(self._prediction.dtype, np.number):
            raise TypeError("prediction could not be converted to a Numpy array")
        if not np.issubdtype(self._truth.dtype, np.number):
            raise TypeError("truth could not be converted to a Numpy array")

        # Raise ValueError if the arrays have different shapes
        if self._prediction.shape != self._truth.shape:
            raise ValueError("prediction and truth arrays do not have the same dimensions")

    def _calculate_metric(self):
        """
        Calculate the Brier metric for the input data.
        """

        # If inputs are 1-d arrays adjust axis and retained output dimensions.
        # Note: It's safe to check just self._prediction, becuase in
        # _validate_data, we raise an exception if _prediction and _truth do not
        # have the same shape.
        AVERAGE_ALONG_ROWS = 1
        KEEP_DIMS = False
        if self._prediction.ndim == 1:
            AVERAGE_ALONG_ROWS = 0
            KEEP_DIMS = True

        return np.mean(
            (self._prediction - self._truth)**2,
            axis=AVERAGE_ALONG_ROWS,
            keepdims=KEEP_DIMS)
