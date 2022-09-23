import numpy as np
from .base import MetricEvaluator

class Brier(MetricEvaluator):
    """ Brier score """
    def __init__(self, prediction, truth):
        """ Class constructor """
        super().__init__(None)

        self._prediction = np.array(prediction)
        self._truth = np.array(truth)

    def evaluate(self) -> float:
        """
        Evaluate the Brier score

        Parameters
        prediction: numpy.ndarray, float
            predicted class probabilities
        truth: numpy.ndarray, int
            true classes mask

        

        Based on https://en.wikipedia.org/wiki/Brier_score#Original_definition_by_Brier
        """
        
        return np.average((self._prediction - self._truth)**2)