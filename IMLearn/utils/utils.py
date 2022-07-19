from typing import Tuple
import numpy as np
import pandas as pd

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    train_amount = int(np.ceil(len(y) * train_proportion))
    rows_for_train = np.random.choice(np.arange(len(y)), train_amount, replace=False)
    rows_for_test = [i for i in range(len(y)) if i not in rows_for_train]
    train_X = X.iloc[rows_for_train]
    test_X = X.iloc[rows_for_test]
    train_Y = y.iloc[rows_for_train]
    test_Y = y.iloc[rows_for_test]
    return train_X, train_Y, test_X, test_Y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    classes = np.unique(a)
    mat = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            mat[i][j] = np.sum((a == classes[i]) & (b == classes[j]))
    return mat
