from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    sets = np.remainder(np.arange(y.size), cv)
    sets = np.random.permutation(sets)
    train_scores = list()
    validation_scores = list()
    for i in range(cv):
        test_X = X[sets == i]
        test_y = y[sets == i]
        train_X = X[sets != i]
        train_Y = y[sets != i]
        h_i_theta = estimator.fit(train_X, train_Y)
        train_scores.append(scoring(h_i_theta.predict(train_X), train_Y))
        validation_scores.append(scoring(h_i_theta.predict(test_X), test_y))

    return np.sum(train_scores) / cv, np.sum(validation_scores) / cv

