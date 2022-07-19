import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IMLearn.utils.utils import confusion_matrix
pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images("C:\HUJI_computer_projects\IML\IML_HUJI\datasets\mnist-train-images.gz"),
            load_labels('C:\HUJI_computer_projects\IML\IML_HUJI\datasets\mnist-train-labels.gz'),
            load_images('C:\HUJI_computer_projects\IML\IML_HUJI\datasets\mnist-test-images.gz'),
            load_labels('C:\HUJI_computer_projects\IML\IML_HUJI\datasets\mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray")\
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False)\
        .update_xaxes(showticklabels=False)\
        .update_yaxes(showticklabels=False)

def get_callback(**kwargs):
    values = list()
    grad_norm = list()

    def callback(**kwargs):
        values.append(kwargs["val"])
        grad_norm.append(np.linalg.norm(kwargs["weights"]))

    return callback, values, grad_norm

if __name__ == '__main__':
    callback, values, grad_norm = get_callback()
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    hidden_size = 64
    relu1 = ReLU()
    relu2 = ReLU()
    lr = FixedLR(0.1)
    loss = CrossEntropyLoss()
    layer_one = FullyConnectedLayer(input_dim=len(train_X[0]), output_dim=hidden_size, activation=relu1,
                                    include_intercept=True)
    hidden_one = FullyConnectedLayer(input_dim=hidden_size, output_dim=hidden_size, activation=relu2,
                                     include_intercept=True)
    layer_two = FullyConnectedLayer(input_dim=hidden_size, output_dim=10, include_intercept=True)
    gradient = StochasticGradientDescent(learning_rate=lr, max_iter=10000, batch_size=256, callback=callback)
    nn = NeuralNetwork(modules=[layer_one, hidden_one, layer_two], loss_fn=loss, solver=gradient)
    nn.fit(train_X, train_y)
    # print("loss ex 5", nn._loss(test_X, test_y))
    #
    # # Plotting convergence process
    # import matplotlib.pyplot as plt
    # plt.title(f"Loss as function of iteration_{hidden_size}")
    # plt.xlabel("iteration")
    # plt.ylabel("loss")
    # plt.scatter(list(range(len(values))), values)
    # plt.show()
    # plt.clf()
    #
    # plt.title(f"Gradient Norm as function of iteration_{hidden_size}")
    # plt.xlabel("iteration")
    # plt.ylabel("grad norm")
    # plt.scatter(list(range(len(values))), grad_norm)
    # plt.show()
    # plt.clf()
    # # Plotting test true- vs predicted confusion matrix
    # pred_y = nn.predict(test_X)
    # confusion_mat = confusion_matrix(pred_y, test_y)
    # print(np.array_str(confusion_mat, precision=2))
    #
    # # ---------------------------------------------------------------------------------------------#
    # # Question 8: Network without hidden layers using SGD                                          #
    # # ---------------------------------------------------------------------------------------------#
    # layer_one = FullyConnectedLayer(input_dim=len(train_X[0]), output_dim=10, activation=relu1,
    #                                 include_intercept=True)
    # callback2, values, grad_norm = get_callback()
    # gradient = StochasticGradientDescent(learning_rate=lr, max_iter=10000, batch_size=256, callback=callback2)
    # nn = NeuralNetwork(modules=[layer_one], loss_fn=loss, solver=gradient)
    # nn.fit(train_X, train_y)
    # print("loss ex 8", nn._loss(test_X, test_y))


    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#

    test_X_7 = test_X[7:]
    pred_7 = nn.compute_prediction(test_X_7)
    seven_sorted_pics = np.argsort(np.max(pred_7, axis=1))
    best = plot_images_grid(test_X_7[seven_sorted_pics[:64], :])
    worst = plot_images_grid(test_X_7[seven_sorted_pics[-64:], :])
    import plotly.offline

    plotly.offline.plot(best, filename=fr"C:\HUJI_computer_projects\IML\ex7\data\best7.html")
    plotly.offline.plot(worst, filename=fr"C:\HUJI_computer_projects\IML\ex7\data\worst7.html")

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    raise NotImplementedError()
