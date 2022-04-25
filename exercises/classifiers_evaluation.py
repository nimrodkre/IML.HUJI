import os.path

from math import atan2, pi
import pandas as pd
from IMLearn.metrics import accuracy
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from matplotlib.colors import ListedColormap
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt



def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data_path = os.path.join(__file__.split("exercises")[0], "datasets", f)
        X, y = load_dataset(data_path)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def callback(preceptron, dummy1, dummy2):
            losses.append(preceptron.loss(X, y))
        Perceptron(callback=callback, include_intercept=True).fit(X, y)

        iterations = [i+1 for i in range(len(losses))]
        plt.plot(iterations, losses)
        plt.title(f"{n} Loss as function of iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.show()
        plt.clf()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return plt.plot(mu[0] + xs, mu[1] + ys, marker="_", color="black")

def __draw_X(mu):
    for i in range(len(mu)):
        plt.plot(mu[i], marker="x")

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data_path = os.path.join(__file__.split("exercises")[0], "datasets", f)
        X, y = load_dataset(data_path)

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()
        lda.fit(X, y)
        gnb.fit(X, y)
        colors = ("green", "blue", "red")


        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        classifiers = [lda, gnb]
        for classifier in classifiers:
            X1, X2 = np.meshgrid(np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
                                 np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01))
            plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                         alpha=0.75, cmap=ListedColormap(colors))
            for i, j in enumerate(np.unique(y)):
                plt.scatter(X[y == j, 0], X[y == j, 1],
                            c=ListedColormap(colors)(i), label=j)
            y_pred = classifier.predict(X)
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            plt.plot()
            plt.legend()
            for i in range(len(np.unique(y))):
                if classifier == lda:
                    plt.title(f"LDA accuracy {accuracy(y, y_pred)}")
                    get_ellipse(classifier.mu_[i], classifier.cov_)
                else:
                    plt.title(f"Gaussian accuracy {accuracy(y, y_pred)}")
                    get_ellipse(classifier.mu_[i], np.array([[gnb.vars_[i][0], 0], [0, gnb.vars_[i][1]]]))
                plt.scatter(classifier.mu_[i][0], classifier.mu_[i][1], marker="x", color="black")
            plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

