# Non-Linear regression model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# Todo: Importing data
df = pd.read_csv("../data/china_gdp.csv")
print(df.head(5))

# Todo: plotting the data set
plt.figure(figsize=(8, 5))
x_data = df["Year"].values
y_data = df["Value"].values

plt.plot(x_data, y_data, 'bo')
plt.xlabel("Year")
plt.ylabel("GDP")
plt.show()

# Todo: Choosing a model
# From the initial look at the plot, we determine that the logistic function
# could be a good approaximation, since it has a property of starting with a
# slow growth, increasing growth in the middle, and then decreasing again
# at the end; as illustrated below:
# x = np.arange(-5.0, 5.0, 0.1)
# y = 1.0 / (1.0 + np.exp(-x))
#
# plt.plot(x, y, 'r')
# plt.xlabel("Independent Variable")
# plt.ylabel("Dependent Variable")
# plt.show()


# Todo: Building the Model
# Build regression model and initialize its parameters
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y


# # Look at the sample sigmoid line that might fit with the data
# beta_1 = 0.10
# beta_2 = 1990
#
# # Todo: Logistic function
# y_pred = sigmoid(x_data, beta_1, beta_2)
# # print(y_pred)
#
# # Todo: plot initial prediction against datapoints
# plt.plot(x_data, y_pred*15000000000000., 'r')
# plt.plot(x_data, y_data, 'bo')
# plt.show()

# Todo: Find the best parameters for the model.
# we can use curve_fit which uses nonlinear least squares to fit our sigmoid function, to data.
# Optimal values for the parameters so that the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.

# first normalize data: x and y
x_norm = x_data/max(x_data)
y_norm = y_data/max(y_data)

# print(x_norm, y_norm)

# estimate parameters: beta_1, beta_2
# popt, pcov = curve_fit(sigmoid, x_norm, y_norm)
# beta_1 = popt[0]
# beta_2 = popt[1]
#
# # print the final parameters
# print("\u03B2\u2081= {:.5f}, \u03B2\u2082= {:.5f}".format(beta_1, beta_2))
#
#
# # Todo: plot the resulting regression model
# x = np.linspace(1990, 2015, 55)
# x = x/max(x)  # normalize x
# plt.figure(figsize=(8, 5))
# y = sigmoid(x, beta_1, beta_2)
# plt.plot(x_norm, y_norm, 'bo', label="Data sample")
# plt.plot(x, y, 'r', linewidth=3.0, label="Regression fit")
# plt.legend(loc="best")
# plt.xlabel("Independent Variable")
# plt.ylabel("Dependent Variable")
# plt.show()


# Todo: Model Evaluation
# Todo: split data into train/test
msk = np.random.rand(len(df)) < 0.8

# Training set
train_x = x_norm[msk]
train_y = y_norm[msk]

# Test set
test_x = x_norm[~msk]
test_y = y_norm[~msk]

# Todo: build the model using train data
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# print the final parameters
print("\u03B2\u2081= {:.5f}, \u03B2\u2082= {:.5f}".format(popt[0], popt[1]))

# Todo: predict using test data
y_hat = sigmoid(test_x, *popt)

# Todo: evalution
print("Mean absolute error: {:.2f}".format(np.mean(np.absolute(y_hat - test_y))))
print("Residual sum of squares (MSE): {:.2f}".format(np.mean((y_hat - test_y) ** 2)))
print("R2-score: {:.2f}".format(r2_score(y_hat, test_y)))






