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
        # sample = np.random.choice(X, num, replace=False)
        diff = np.abs(np.mean(sample) - 10)
        sample_diff.append(diff)

    plt.scatter(divisible_by_10, sample_diff)
    plt.title("Diff Between Expected Value of All Samples and N Samples")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Distance From the Original Expected Value")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    y = univariateGaussian.pdf(X)
    plt.clf()
    plt.scatter(X, y, s=2)
    plt.title("PDF Over Original Data")
    plt.xlabel("Sample")
    plt.ylabel("PDF Sample Value")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, 1000)
    multivariateGaussian = MultivariateGaussian()
    multivariateGaussian.fit(X)
    print(multivariateGaussian.mu_)
    print(multivariateGaussian.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    heatmap = np.zeros([200, 200])
    from tqdm import tqdm
    for i in tqdm(range(len(f1))):
        for j in range(len(f3)):
            heatmap[i][j] = multivariateGaussian.log_likelihood(np.array([f1[i], 0, f3[j], 0]).reshape(1, 4), sigma, X)
    plt.pcolormesh(f3, f1, heatmap)
    plt.title("Log Likelihood of Expected Value [f1, 0, f3, 0]")
    plt.xlabel("f3")
    plt.ylabel("f1")
    plt.show()
    # Question 6 - Maximum likelihood
    max_coordinates = np.where(heatmap == np.amax(heatmap))

    print(f"({round(f1[max_coordinates[0]][0], 3)}, {round(f3[max_coordinates[1]][0], 3)})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()