import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
import plotly.offline
import matplotlib.pyplot as plt
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import misclassification_error
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = list()
    weigths = list()
    def callback(**kwargs):
        values.append(kwargs["val"])
        weigths.append(kwargs["weights"])
    return callback, values, weigths


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    callback, values, weights = get_gd_state_recorder_callback()
    for regularizer in [L1, L2]:
        for eta in etas:
            reg = regularizer(init)
            gradient = GradientDescent(FixedLR(eta), callback=callback)
            gradient.fit(reg, None, None)
            path = np.stack(weights, axis=0)
            fig = plot_descent_path(regularizer, path, f"{type(regularizer)}, {eta}")
            if regularizer is L1:
                plotly.offline.plot(fig,
                                    filename=f"C:\HUJI_computer_projects\IML\ex6\pickeling\check_type_L1_{eta}.html")
            else:
                plotly.offline.plot(fig,
                                    filename=f"C:\HUJI_computer_projects\IML\ex6\pickeling\check_type_L2_{eta}.html")
            plt.scatter(list(range(len(values))), values)
            if regularizer is L1:
                plt.savefig(fr"C:\HUJI_computer_projects\IML\ex6\pickeling\values_type_L1_{eta}.jpg")
            else:
                plt.savefig(fr"C:\HUJI_computer_projects\IML\ex6\pickeling\values_type_L2_{eta}.jpg")
            plt.clf()
            values.clear()
            weights.clear()




def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    for gamma in gammas:
        reg = L1(init)
        lt = ExponentialLR(eta, gamma)
        callback, values, weights = get_gd_state_recorder_callback()
        gradient = GradientDescent(lt, callback=callback)
        gradient.fit(reg, None, None)
        plt.scatter(list(range(len(values))), values, label=gamma)
    plt.legend()
    plt.savefig(fr"C:\HUJI_computer_projects\IML\ex6\pickeling\values_exponential_gamma.jpg")
    plt.clf()

    reg = L1(init)
    lt = ExponentialLR(eta, 0.95)
    callback, values, weights = get_gd_state_recorder_callback()
    gradient = GradientDescent(lt, callback=callback)
    gradient.fit(reg, None, None)
    fig = plot_descent_path(L1, np.array(weights), f"{type(reg)}, {eta}")
    plotly.offline.plot(fig,
                        filename=f"C:\HUJI_computer_projects\IML\ex6\pickeling\exp_path.html")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(r"C:\HUJI_computer_projects\IML\IML_HUJI\datasets\SAheart.data")
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    y_train = y_train.to_numpy()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    logistic_regression = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000))
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict_proba(X_train)
    false_positive_rate = list()
    true_positive_rate = list()
    # Plotting convergence rate of logistic regression over SA heart disease data
    from tqdm import tqdm
    for alpha in tqdm(range(0, 101)):
        y_alpha = (y_pred > alpha * 0.01).astype(int)
        true_positive = np.sum(np.logical_and(y_alpha, y_alpha == y_train))
        true_negative = np.sum(np.logical_and(np.logical_not(y_alpha), y_alpha == y_train))
        false_positive = np.sum(np.logical_and(y_alpha, y_alpha != y_train))
        false_negative = np.size(y_alpha) - true_positive - true_negative - false_positive
        if false_positive == 0:
            false_positive_rate.append(false_positive)
        else:
            false_positive_rate.append(false_positive / (false_positive + true_negative))
        if true_positive == 0:
            true_positive_rate.append(0)
        else:
            true_positive_rate.append(true_positive / (false_negative + true_positive))
    plt.scatter(false_positive_rate, true_positive_rate)
    plt.title("ROC curve")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.show()

    best_alpha_index = np.argmax(np.array(true_positive_rate) - np.array(false_positive_rate))
    best_alpha = 0.01 * best_alpha_index
    print("Best Alpha", best_alpha)
    logistic_regression.alpha_ = best_alpha
    best_alpha_loss = logistic_regression.loss(X_test, y_test)
    print("Best alpha loss", best_alpha_loss)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in ["l1", "l2"]:
        validation_scores = list()
        for lam in lambdas:
            log = LogisticRegression(penalty=penalty, alpha=0.5, lam=lam)
            train_score, validation_score = cross_validate(log, X_train, y_train, misclassification_error)
            validation_scores.append(validation_score)
        min_score = min(validation_scores)
        min_index = validation_scores.index(min_score)
        min_lambda = lambdas[min_index]
        log = LogisticRegression(penalty=penalty, alpha=0.5, lam=min_lambda)
        log.fit(X_train, y_train)
        err = log.loss(X_test, y_test)
        print("penalty", penalty, "Best lambda", min_lambda)
        print("penalty", penalty, "Best err", err)

if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
