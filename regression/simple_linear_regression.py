# Implementation of linear regression using python numpy

# Importing required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# create regression model
def linear_regression(x, y, title=None, x_label=None, y_label=None, filename=None, show=True):
    """
    Simple Linear Regression Model using ordinary least squares (OLS).
    Fits regression line. Returns intercept and coefficient.
    :param x: Independent variables
    :param y: Dependent variable
    :param title: title of the plot
    :param x_label: x-label or independent variable
    :param y_label: y-label or dependent variable
    :param filename: filename to save figure
    :param show: It true shows current figure (default True)
    :return: simple linear regression model
    """

    # Estimate the coefficients
    b = estimate_coef(x, y)
    # print("Intercept: b_0 = {}".format(b[0]))
    # print("Coefficients: b_1 = {}".format(b[1]))

    # Plot the regression line
    fit_regression_line(x, y, b, title, x_label, y_label, filename, show)
    return b


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


def fit_regression_line(x, y, b, title, x_label, y_label, filename, show):

    # plotting the observational points as scatter plot
    plt.scatter(x, y, color="b", marker="o", s=30)

    # predicted response vector y_hat
    y_hat = b[0] + b[1]*x

    # plo the regression line
    plt.plot(x, y_hat, color='r')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show(block=False)
        plt.pause(3)
        plt.close()


# Evaluating the performance of the model
# We will be using Root mean squared error(RMSE) and Coefficient of Determination(R² score) to evaluate our model.

# Calculate Residual sum of squares (MSE)
def get_mse(y_hat, y_actual):
    """
    RMSE is the square root of the average of the sum of the squares of residuals.
    :param y_hat:
    :param y_actual:
    :return:
    """
    mse = np.mean((y_hat - y_actual)**2)

    return mse


# Calculate coefficient of determination: r2 score or r-quare
def r2_score(y_hat, y_actual):
    """
    R² score or the coefficient of determination explains
    how much the total variance of the dependent variable
    can be reduced by using the least square regression.
    :return:
    """
    # calculate sum of squares of residuals
    ss_res = np.sum((y_actual - y_hat)**2)

    # calculate total sum of squares
    ss_tot = np.sum(y_actual - np.mean(y_actual)**2)

    # calculate r2 score
    r2 = 1 - (np.absolute(ss_res/ss_tot))

    return r2


# predict the value of dependent variable y_pred
# from independent variable x using this regression model
def predict(x, b):

    # predicted response vector y_hat
    y_pred = b[0] + b[1]*x

    return y_pred


if __name__ == "__main__":

    # Reading the data in
    df = pd.read_csv("../data/FuelConsumption.csv")
    # print(df.head())
    # print(df.columns)

    # Select some features to explore more
    # Features taken: 'ENGINESIZE', 'CO2EMISSIONS'
    cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]
    # print(cdf.head(10))

    # plot features using scatterplot
    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Creating train and test dataset: Train/Test Split
    # Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for
    # testing. We create a mask to select random rows using np.random.rand() function:
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    # Train data distribution
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    # Modeling
    # traning set
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    coeff = linear_regression(train_x, train_y)

    # Get the coefficient and y-intercept
    print("Coefficient and y-intercept:")
    print("Coefficient: {:.2f}".format(coeff[1]))
    print("Intercept: {:.2f}".format(coeff[0]))

    # There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our
    # model based on the test set:
    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_ = predict(test_x, coeff)

    # Calculate evaluation metrics
    print("\nEvaluation metrics:")
    print("Mean absolute error: {:.2f}".format(np.mean(np.absolute(test_y_ - test_y))))
    print("Residual sum of squares (MSE): {:.2f}".format(get_mse(test_y_, test_y)))
    print("R2-score: {:.2f}".format(r2_score(test_y_, test_y)))



