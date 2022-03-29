import matplotlib.pyplot as plt

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
    df = df[pd.to_datetime(df.Date).dt.year == df.Year]
    df = df[pd.to_datetime(df.Date).dt.month == df.Month]
    df = df[pd.to_datetime(df.Date).dt.day == df.Day]
    df["DayOfYear"] = pd.to_datetime(df.Date).dt.day_of_year
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"C:\HUJI_computer_projects\IML\IML_HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df.Country == "Israel"]
    israel_df = israel_df[israel_df.Temp > -70]
    temp_df = israel_df.Temp
    israel_day_of_year = israel_df.DayOfYear
    year_colors = {1995: "red", 1996: "sienna", 1997: "darkorange",
                   1998: "gold", 1999: "yellow", 2000: "lawngreen",
                   2001: "limegreen", 2002: "springgreen", 2003: "turquoise",
                   2004: "teal", 2005: "cyan", 2006: "skyblue", 2007: "slategray"}
    plt.scatter(israel_df.DayOfYear, israel_df.Temp, c=israel_df.Year.map(year_colors))
    plt.title("Temperature as Function of Day of Year")
    plt.ylabel("Temperature")
    plt.xlabel("Day of Year")
    plt.show()
    plt.clf()

    israel_df_month_temp = israel_df[["Temp", "Month"]]
    month_std = israel_df_month_temp.groupby(["Month"]).agg("std")
    plt.bar([i for i in range(1, len(month_std) + 1)], month_std.Temp)
    plt.title("Bar Graph of STD as Function of Month")
    plt.xlabel("Month")
    plt.ylabel("STD")
    plt.show()
    plt.clf()


    # Question 3 - Exploring differences between countries
    month_country_temp_df = df.drop(["Year", "Day", "DayOfYear"], axis=1)
    std = month_country_temp_df.groupby(["Month", "Country"]).agg("std")
    mean = month_country_temp_df.groupby(["Month", "Country"]).agg("mean")
    countries_colors = {"South Africa": "blue", "Jordan": "green",
                        "Israel": "yellow", "The Netherlands": "black"}
    for i in range(len(countries_colors)):
        country = mean.iloc[i]._name[1]
        x = list(range(1, 13))
        y = [mean.iloc[i + j * len(countries_colors)].Temp for j in range(12)]
        y_error = [std.iloc[i + j * len(countries_colors)].Temp for j in range(12)]
        plt.errorbar(x, y, yerr=y_error, label=country)
    plt.legend()
    plt.show()
    plt.clf()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()