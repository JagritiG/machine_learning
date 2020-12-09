# Implementation of simple linear regression using scikit-learn

# Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('seaborn-whitegrid')


def linear_regression(df, x, y, x_label=None, y_label=None, filename=None, show=True):
    """
    Simple linear regression uisng least squares.
    Prints coefficient, y-intercept, and evaluation metrics:
    Mean absolute error,  Mean squared error, Residual sum of squares (MSE), R2 (r-square)
    :param df: pandas data frame
    :param x: independent variable (feature)
    :param y: dependent variable
    :param x_label: x_label
    :param y_labe: y_label
    :param filename: filename to save plot
    :param show: If true shows current figure
    :return:
    """

    try:

        # create the figure where our subplots will go
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

        # Creating train and test dataset: Train/Test Split
        # Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for
        # testing. We create a mask to select random rows using np.random.rand() function:
        msk = np.random.rand(len(df)) < 0.8
        train = cdf[msk]
        test = cdf[~msk]

        # Plot train data distribution
        axes2 = fig.add_subplot(3, 1, 2)
        axes2.scatter(train[x], train[y], color='blue', label='Sample data')
        axes2.set_title("Train Data Distribution")
        axes2.set_xlabel(x_label)
        axes2.set_ylabel(y_label)
        axes2.legend()

        # Modeling with training set
        train_x = np.asanyarray(train[[x]])
        train_y = np.asanyarray(train[[y]])

        # Model initialization
        regression_model = LinearRegression()
        regression_model.fit(train_x, train_y)

        # Get the coefficient and y-intercept
        coef = regression_model.coef_
        y_intercept = regression_model.intercept_
        print("Coefficient: {}".format(coef))
        print("Intercept: {}".format(y_intercept))

        # Predict the response, based on test set
        test_x = np.asanyarray(test[[x]])
        test_y = np.asanyarray(test[[y]])
        y_pred = regression_model.predict(test_x)

        # Evaluate the model
        # There are different model evaluation metrics to calculate the accuracy of the model
        # Calculate evaluation metrics
        mae = mean_squared_error(test_y, y_pred)
        rmse = np.mean(np.absolute(y_pred - test_y))
        mse = np.mean((y_pred - test_y) ** 2)
        r2 = r2_score(y_pred, test_y)

        print("\nEvaluation metrics:")
        print("Mean absolute error: {:.2f}".format(mae))
        print("Mean squared error: {:.2f}".format(rmse))
        print("Residual sum of squares (MSE): {:.2f}".format(mse))
        print("R2-score: {:.2f}".format(r2))

        # plot the response with regression line
        axes3 = fig.add_subplot(3, 1, 3)
        axes3.scatter(train[x], train[y], color='blue', label='Sample data')
        axes3.plot(train_x, regression_model.coef_[0][0]*train_x + regression_model.intercept_[0], '-r', label='Regression model')
        axes3.set_title('$R^2= %.2f$' % r2, fontsize=12)
        axes3.set_xlabel(x_label)
        axes3.set_ylabel(y_label)
        axes3.legend()
        axes3.text(6.6, 156, 'Coefficient: {} \nIntercept: {} \nMSE: {:.2f} \nR2-score: {:.2f}'.format(coef[0][0], y_intercept[0], mse, r2),
                   fontweight='semibold', bbox={'facecolor': 'green', 'alpha': 0.2, 'pad': 8})
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        if show:
            plt.show(block=False)
            plt.pause(5)
            plt.close()

    except Exception as e:
        print('Error: {}'.format(str(e)))


if __name__ == "__main__":

    # Reading the data in
    df = pd.read_csv("../data/FuelConsumption.csv")
    # print(df.head())
    # print(df.columns)

    linear_regression(df, "ENGINESIZE", "CO2EMISSIONS", "Engine Size", "Co2 Emission", "../results/EngineSize_vs_Co2Emission_slregr.png")







