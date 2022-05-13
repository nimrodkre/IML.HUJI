from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        num_labels = len(X[0])
        min_error = 100000000
        min_error_threshold = 100000000
        min_label = 0
        for i in range(num_labels):
            thr, thr_err = self._find_threshold(X[:,i], y, 1)
            thr2, thr_err2 = self._find_threshold(X[:, i], y, -1)
            if min(thr_err, thr_err2) < min_error:
                if thr_err2 < thr_err:
                    min_error = thr_err2
                    min_error_threshold = thr2
                    self.sign_ = -1
                else:
                    min_error = thr_err
                    min_error_threshold = thr
                    self.sign_ = 1
                min_label = i
        self.j_ = min_label
        self.threshold_ = min_error_threshold

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        under_over_array = (X[:, self.j_] >= self.threshold_)
        value_array = under_over_array * 2 - 1
        return self.sign_ * value_array

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_values, sorted_labels = zip(*sorted(zip(values, labels)))

        error = self.calc_loss(sorted_labels, [sign] * len(sorted_labels))
        min_error = error
        min_error_threshold = sorted_values[0]
        for i in range(1, len(sorted_values)):
            if np.sign(sorted_labels[i - 1]) == -sign:
                error -= np.abs(sorted_labels[i - 1])
            else:
                error += np.abs(sorted_labels[i - 1])
            if error < min_error:
                min_error = error
                min_error_threshold = sorted_values[i]
        error = self.calc_loss(sorted_labels, [-sign] * len(sorted_labels))
        if error < min_error:
            min_error = error
            min_error_threshold = np.Inf
        return min_error_threshold, min_error

    def calc_loss(self, y, y_pred):
        loss = 0
        for i in range(len(y)):
            if y[i] * y_pred[i] < 0:
                loss += np.abs(y[i])
        return loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_predicted = self.predict(X)
        return misclassification_error(y, y_predicted)
