import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename)
    df = df.dropna(axis=0 ,how="any")
    df = df[df.Country != ""]
    df = df[df.City != ""]
    df = df[df.Year > 0]
    df = df[df.Day > 0]
    df = df[df.Day <= 31]
    df = df[df.Month > 0]
    df = df[df.Month <= 12]
    df = df[df.Date.dt.year == df.Year]
    df = df[df.Date.dt.month == df.Month]
    df = df[df.Date.dt.day == df.Day]
    df["DayOfYear"] = df.Date.df.dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"C:\HUJI_computer_projects\IML\IML_HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()