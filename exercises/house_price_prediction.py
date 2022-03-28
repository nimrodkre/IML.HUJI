from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from IMLearn.utils.utils import split_train_test

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio
pio.templates.default = "simple_white"
import pandas as pd
import os
import random


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.dropna(axis=0 ,how="any")
    df = df[df.price > 0]
    df = df[df.sqft_living > 0]
    df = df[df.sqft_lot > 0]
    df = df.drop(labels=["id"], axis=1)
    df = build_date(df)
    df = build_zipcodes(df)
    return df

def build_date(df):
    years = list()
    months = list()
    days = list()
    for i in range(len(df)):
        row = df.iloc[i]
        years.append(get_year(row.date))
        months.append(get_month(row.date))
        days.append(get_day(row.date))
    df["date_year"] = years
    df["date_month"] = months
    df["date_day"] = days
    return df.drop(labels=["date"], axis=1)

def build_zipcodes(df):
    return pd.get_dummies(df, columns=["zipcode"])

def get_year(date):
    return int(date[:4])

def get_month(date):
    return int(date[4:6])

def get_day(date):
    return int(date[6:8])

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for column_name in X:
        column_data = X[column_name]
        data_cov = np.cov(column_data, y)[0][1]
        column_std = np.std(column_data)
        y_std = np.std(y)
        pearson_correlation = data_cov / (y_std * column_std)
        plt.scatter(column_data, y)
        plt.suptitle(f"{y._name} as {column_name}, Pearson Correlation: {pearson_correlation}",
                     fontsize=10)
        plt.xlabel(column_name)
        plt.ylabel(y._name)
        plt.rc('font', size=8)
        plt.savefig(os.path.join(output_path, f"{y._name} as {column_name}"))
        plt.clf()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df = load_data(r"C:\HUJI_computer_projects\IML\IML_HUJI\datasets\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(df, df.price)

    # Question 3 - Split samples into training- and testing sets.
    data_no_price = df.drop(labels=["price"], axis=1)
    train_X, train_Y, test_X, test_Y = split_train_test(data_no_price, df.price, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    linear_regression = LinearRegression()
    errors = list()
    means = list()
    from tqdm import tqdm
    percentages = [p for p in range(10, 101)]
    for p in tqdm(percentages):
        mean_loss = list()
        for i in range(10):
            num_samples = int(len(train_X) * p / 100)
            rows = random.sample(np.arange(0, len(train_X)).tolist(), num_samples)
            train_x_fraction = train_X.iloc[rows,]
            train_y_fraction = train_Y.iloc[rows,]
            linear_regression.fit(train_x_fraction, train_y_fraction)
            mean_loss.append(linear_regression.loss(test_X, test_Y))
        means.append(np.mean(mean_loss))
        errors.append(2 * np.std(mean_loss))
    plt.errorbar(percentages, means, errors)
    plt.show()
