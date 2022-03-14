from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(loc=10, scale=1, size=(1000,))
    univariateGaussian = UnivariateGaussian()
    univariateGaussian.fit(X)
    print(f'({univariateGaussian.mu_}, {univariateGaussian.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    divisible_by_10 = [x for x in range(1001) if x % 10 == 0 and x != 0]
    sample_diff = []
    for num in divisible_by_10:
        sample = X[:num]
        diff = np.abs(np.mean(sample) - univariateGaussian.mu_)
        sample_diff.append(diff)
    # fig = go.Figure(data=go.Scatter(x=divisible_by_10, y=sample_diff, mode='markers'))
    # fig.show()
    plt.scatter(divisible_by_10, sample_diff)
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
