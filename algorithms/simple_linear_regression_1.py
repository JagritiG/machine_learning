# Implementation of linear regression using python numpy

# Importing required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def estimate_coef(x, y):

    # get the number of observations
    n = np.size(x)
    # print(n)

    # mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate Cov[x,y] and Var[x]
    cov_xy = n*np.sum(x*y) - (np.sum(x)*np.sum(y))
    var_x = n*np.sum(x*x) - (np.sum(x)*np.sum(x))

    # Calculate intercept b_0 and coefficient b_1
    b_1 = cov_xy/var_x
    b_0 = y_mean - b_1*x_mean

    return [b_0, b_1]


def fit_regression_line(x, y, b):

    # plotting the observational points as scatter plot
    plt.scatter(x, y, color="b", marker="o", s=30)

    # predicted response vector y_hat
    y_hat = b[0] + b[1]*x

    # plo the regression line
    plt.plot(x, y_hat, color='r')

    plt.title("Simple Linear Regression Model")
    plt.xlabel("Independent Variable")
    plt.ylabel("Dependent Variable")
    plt.show()


if __name__ == "__main__":

    # # Reading the data in
    df = pd.read_csv("../data/FuelConsumption.csv")
    # # print(df.head())
    # # print(df.columns)
    #
    # # Select some features to explore more
    # # Features taken: 'ENGINESIZE', 'CO2EMISSIONS'
    cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]
    # print(cdf.head(10))
    #
    # # observations
    x = cdf["ENGINESIZE"]
    y = cdf['CO2EMISSIONS']
    # # print(x, y)

    # observations
    # x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    # Estimate the coefficients
    b = estimate_coef(x, y)
    b_0 = b[0]
    b_1 = b[1]
    print("Intercept: b_0 = {}".format(b_0))
    print("Coefficients: b_1 = {}".format(b_1))

    # Plot the regression line
    fit_regression_line(x, y, b)



