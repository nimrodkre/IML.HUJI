from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    max_val = 2
    min_val = -1.2
    x = np.linspace(min_val, max_val, n_samples)
    samples_x = x

    samples_y = (x + 3) * (x+ 2) * (x+1) * (x-1) * (x-2) + np.random.randn(n_samples) * noise
    train_X, train_Y, test_X, test_Y = split_train_test(pd.DataFrame(samples_x), pd.DataFrame(samples_y), 2/3)
    train_X = np.array([x[0] for x in train_X.to_numpy()])
    train_Y = np.array([y[0] for y in train_Y.to_numpy()])
    test_X = np.array([x[0] for x in test_X.to_numpy()])
    test_Y = np.array([y[0] for y in test_Y.to_numpy()])
    plt.scatter(train_X, train_Y, color="blue", label="train")
    plt.scatter(test_X, test_Y, color="red", label="test")
    plt.scatter(x, (x + 3) * (x + 2) * (x+1) * (x-1) * (x-2), color="black", label="noiseless")
    plt.xlabel("X")
    plt.ylabel("f(X)")
    plt.legend()
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    errors = list()
    training_scores = list()
    validation_scores = list()
    for degree in range(11):
        polynom_fit = PolynomialFitting(degree)
        train_score, validation_score = cross_validate(polynom_fit, train_X, train_Y, mean_square_error, cv=5)
        training_scores.append(train_score)
        validation_scores.append(validation_score)
    plt.scatter(list(range(11)), training_scores, color="blue", label="training_scores")
    plt.scatter(list(range(11)), validation_scores, color="red", label="validation_scores")
    plt.title("Score as function of the polynomial degree")
    plt.xlabel("polynomial degree")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_scores)
    model_for_best_k = PolynomialFitting(best_k)
    model_for_best_k.fit(train_X, train_Y)
    print("Best K", best_k)
    print("Best k loss", model_for_best_k.loss(test_X, test_Y))



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data, label = datasets.load_diabetes(return_X_y=True)

    training_data = data[:n_samples]
    training_labels = label[:n_samples]
    testing_data = data[n_samples:]
    testing_labels = label[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_range = np.linspace(0, 3, num=n_evaluations)
    ridge_training_scores = list()
    ridge_validation_scores = list()
    lasso_training_scores = list()
    lasso_validation_scores = list()
    from tqdm import tqdm
    for lam in tqdm(lambda_range):
        ridge = RidgeRegression(lam=lam)
        lasso = Lasso(alpha=lam)
        ridge_train_score, ridge_validation_score = cross_validate(ridge, training_data, training_labels, mean_square_error, cv=5)
        lasso_train_score, lasso_validation_score = cross_validate(lasso, training_data, training_labels, mean_square_error, cv=5)
        ridge_validation_scores.append(ridge_validation_score)
        ridge_training_scores.append(ridge_train_score)
        lasso_validation_scores.append(lasso_validation_score)
        lasso_training_scores.append(lasso_train_score)

    plt.scatter(lambda_range, ridge_training_scores, color="blue", label="training")
    plt.scatter(lambda_range, ridge_validation_scores, color="red", label="validation")
    plt.legend()
    plt.title("Ridge scores as function of lambda")
    plt.xlabel("lambda")
    plt.ylabel("score")
    plt.show()

    plt.scatter(lambda_range, lasso_training_scores, color="blue", label="training")
    plt.scatter(lambda_range, lasso_validation_scores, color="red", label="validation")
    plt.legend()
    plt.title("Lasso scores as function of lambda")
    plt.xlabel("lambda")
    plt.ylabel("score")
    plt.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = np.argmin(ridge_validation_scores)
    best_lasso = np.argmin(lasso_validation_scores)

    lasso_best = Lasso(alpha=lambda_range[best_lasso])
    lasso_best.fit(training_data, training_labels)
    lasso_pred = lasso_best.predict(testing_data)
    best_error_lasso = mean_square_error(lasso_pred, testing_labels)

    ridge_best = RidgeRegression(lam=lambda_range[best_ridge])
    ridge_best.fit(training_data, training_labels)
    best_error_ridge = ridge_best._loss(testing_data, testing_labels)

    linear = LinearRegression(include_intercept=True)
    linear.fit(training_data, training_labels)
    best_error_linear = linear.loss(testing_data, testing_labels)

    print("Best ridge error", best_error_ridge, "lambda", lambda_range[best_ridge])
    print("Best lasso error", best_error_lasso, "lambda", lambda_range[best_lasso])
    print("Best linear error", best_error_linear)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
