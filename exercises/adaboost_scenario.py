import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def __create_grid(ada, train_X, test_X, test_y, t, noise, accuracy=0):
    def predict(X):
        return ada.partial_predict(X, t)
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T \
           + np.array([-.1, .1])
    fig = go.Figure(layout=go.Layout(title=rf"iteration: {t}, accuracy:{accuracy}"))
    contour = decision_surface(predict, lims[0], lims[1], showscale=False)
    fig.add_trace(contour)
    symbols = np.array(["circle", "x", "x"])
    fig.add_trace(go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                             marker=dict(color=test_y, symbol=symbols[test_y.astype(np.int32) + 1],
                                         colorscale=[custom[0], custom[-1]]
                                         )))

    import plotly.offline
    plotly.offline.plot(fig, filename=f"C:\HUJI_computer_projects\IML\ex4\pickeling\check_{t}_noise_{noise}.html")

def q4(ada, train_X, test_X, train_y, t, noise, accuracy=0):
    def predict(X):
        return ada.partial_predict(X, t)
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T \
           + np.array([-.1, .1])
    fig = go.Figure(layout=go.Layout(title=rf"iteration: {t}, accuracy:{accuracy}"))
    contour = decision_surface(predict, lims[0], lims[1], showscale=False)
    fig.add_trace(contour)
    symbols = np.array(["circle", "x", "x"])
    scores = ada.D_ / np.max(ada.D_) * 5
    if noise == 0:
        fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y, symbol=symbols[train_y.astype(np.int32) + 1],
                                             colorscale=[custom[0], custom[-1]], size=scores * 5
                                             )))
    else:
        fig.add_trace(go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=train_y, symbol=symbols[train_y.astype(np.int32) + 1],
                                             colorscale=[custom[0], custom[-1]], size=scores * 2
                                             )))

    import plotly.offline
    plotly.offline.plot(fig, filename=f"C:\HUJI_computer_projects\IML\ex4\pickeling\check_{t}_noise_{noise}_diff.html")


def q1(ada, n_learners, test_X, test_y, train_X, train_y):
    test_losses = []
    train_losses = []
    for T in range(1, n_learners + 1):
        test_losses.append(ada.partial_loss(test_X, test_y, T))
        train_losses.append(ada.partial_loss(train_X, train_y, T))

    plt.scatter(list(range(1, n_learners + 1)), test_losses, c="blue", label="test", s=1)
    plt.scatter(list(range(1, n_learners + 1)), train_losses, c="red", label="train", s=1)
    plt.title("Loss as function of number of models")
    plt.ylabel("loss")
    plt.xlabel("number of models")
    plt.legend()
    plt.show()

def q2(ada, train_X, test_X, test_y, noise):
    T = [5, 50, 100, 250]
    for t in T:
        __create_grid(ada, train_X, test_X, test_y, t, noise)

def q3(ada, n_learners, train_X, test_X, test_y, noise):
    min_error = 1
    min_t = 0
    for t in range(1, n_learners + 1):
        err = ada.partial_loss(test_X, test_y, t)
        if err < min_error:
            min_error = err
            min_t = t
    __create_grid(ada, train_X, test_X, test_y, min_t, noise, accuracy=(1 - min_error))

def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    ada = AdaBoost(DecisionStump, n_learners)
    ada.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    q1(ada, n_learners, test_X, test_y, train_X, train_y)

    # Question 2: Plotting decision surfaces
    q2(ada, train_X, test_X, test_y, noise)

    # Question 3: Decision surface of best performing ensemble
    q3(ada, n_learners, train_X, test_X, test_y, noise)

    # Question 4: Decision surface with weighted samples
    q4(ada, train_X, test_X, train_y, 250, noise)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
