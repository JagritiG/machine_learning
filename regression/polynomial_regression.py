# Implementation of Polynomial Regression using scikit-learn

# Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


def polynomial_regression(df, x, y, degree=2, x_label=None, y_label=None, filename=None, show=True):
    """
    Polynomial regression uisng least squares.
    Prints coefficient, y-intercept, and evaluation metrics: R2 (r-square)
    :param df: pandas data frame
    :param x: independent variables (features)
    :param y: dependent variable
    :param x_label: x_label
    :param y_labe: y_label
    :param filename: filename to save plot
    :param show: If true shows current figure
    :return:
    """

    # Todo: create the figure where our subplots will go
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=[10, 8])

    # subplot has 3 rows and 1 column, plot location 1
    # plot features using scatterplot
    axes1 = fig.add_subplot(3, 1, 1)
    axes1.scatter(df[x], df[y], color='blue', label='Sample data')
    axes1.set_title("Data Distribution Before Train/Test Split")
    axes1.set_xlabel(x_label)
    axes1.set_ylabel(y_label)
    axes1.legend()

    # select the feature
    cdf = df[[x, y]]

    # Todo: Creating train and test dataset: Train/Test Split
    # Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for
    # testing. We create a mask to select random rows using np.random.rand() function:
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    # Train data distribution
    axes2 = fig.add_subplot(3, 1, 2)
    axes2.scatter(train[x], train[y], color='blue', label='Sample data')
    axes2.set_title("Train data distribution")
    axes2.set_xlabel(x_label)
    axes2.set_ylabel(y_label)
    axes2.legend()

    # Todo: Polynomial regression
    # PolynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original
    # feature set. That is, a matrix will be generated consisting of all polynomial combinations of the
    # features with degree less than or equal to the specified degree.

    # Modeling with training set
    train_x = np.asanyarray(train[[x]])
    train_y = np.asanyarray(train[[y]])

    # Todo: Polynomial transformation
    # fit_transform takes our x values, and output a list of our data raised from power of 0 to power
    # of N-degree (N = 2, 3).
    poly_model = PolynomialFeatures(degree=degree)
    train_x_poly = poly_model.fit_transform(train_x)
    # print(train_x_poly)

    # Todo: Linear regression model
    lr_model = linear_model.LinearRegression()
    train_y_ = lr_model.fit(train_x_poly, train_y)

    # Get the coefficient and y-intercept
    print("Coefficient and y-intercept:")
    print("Coefficient: {}".format(lr_model.coef_))
    print("Intercept: {}".format(lr_model.intercept_))

    # Todo: Evalualtion
    # Predict the response, based on test set
    test_x = np.asanyarray(test[[x]])
    test_y = np.asanyarray(test[[y]])
    test_x_poly = poly_model.fit_transform(test_x)
    test_y_ = lr_model.predict(test_x_poly)

    # There are different model evaluation metrics to calculate the accuracy of the model
    # Calculate evaluation metrics R2
    r2 = r2_score(test_y_, test_y)

    print("R2-score: {:.2f}".format(r2))

    # plot the model
    axes3 = fig.add_subplot(3, 1, 3)
    axes3.scatter(train[x], train[y], color='blue', label='Sample data')
    xx = np.arange(0.0, 10.0, 0.1)
    if degree == 2:
        yy = lr_model.intercept_[0] + lr_model.coef_[0][1]*xx + lr_model.coef_[0][2]*np.power(xx, 2)
        axes3.plot(xx, yy, '-r', label='Regression model')
        axes3.set_title("\nR2= {:.2f}".format(r2), fontsize=16)
        axes3.set_xlabel(x_label)
        axes3.set_ylabel(y_label)
        axes3.legend()
        axes3.text(6.6, 156, 'Coefficient: {} \nIntercept: {} \nR2-score: {:.2f}'.format(lr_model.coef_[0][0], lr_model.coef_[0], r2),
                   fontweight='semibold', bbox={'facecolor': 'green', 'alpha': 0.2, 'pad': 8})
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        if show:
            plt.show(block=False)
            plt.pause(3)
            plt.close()
    if degree == 3:
        yy = lr_model.intercept_[0] + lr_model.coef_[0][1]*xx + lr_model.coef_[0][2]*np.power(xx, 2) + \
             lr_model.coef_[0][3]*np.power(xx, 3)
        axes3.plot(xx, yy, '-r', label='Regression model')
        axes3.title('$R^2= %.2f$' % r2, fontsize=12)
        axes3.set_set_xlabel(x_label)
        axes3.set_ylabel(y_label)
        axes3.legend()
        axes3.text(6.6, 156, 'Coefficient: {} \nIntercept: {} \nR2-score: {:.2f}'.format(lr_model.coef_[0][0], lr_model.coef_[0], r2),
                   fontweight='semibold', bbox={'facecolor': 'green', 'alpha': 0.2, 'pad': 8})
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        if show:
            plt.show(block=False)
            plt.pause(3)
            plt.close()


if __name__ == "__main__":

    # Reading the data in
    df = pd.read_csv("../data/FuelConsumption.csv")
    # print(df.head())
    # print(df.columns)

    polynomial_regression(df, "ENGINESIZE", "CO2EMISSIONS", 2, "Engine Size", "Co2 Emission", "../results/EngineSize_vs_Co2Emission_polyregr.png")


