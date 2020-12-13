# Implementation of linear regression using scikit-learn

# Importing required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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

# Model initialization
regression_model = LinearRegression()
regression_model.fit(train_x, train_y)

# Get the coefficient and y-intercept
print("Coefficient and y-intercept:")
print("Coefficient: ".format(regression_model.coef_))
print("Intercept: ".format(regression_model.intercept_))


# plot the output
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regression_model.coef_[0][0]*train_x + regression_model.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show(block=False)
plt.pause(3)
plt.close()

# Predict the response, based on test set
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
y_pred = regression_model.predict(test_x)


# Evaluate the model
# There are different model evaluation metrics to calculate the accuracy of the model
# Calculate evaluation metrics
print("\nEvaluation metrics:")
print("Mean absolute error: {:.2f}".format(mean_squared_error(test_y, y_pred)))
print("Mean squared error: {:.2f}".format(np.mean(np.absolute(y_pred - test_y))))
print("Residual sum of squares (MSE): {:.2f}".format(np.mean((y_pred - test_y) ** 2)))
print("R2-score: {:.2f}".format(r2_score(y_pred, test_y)))

