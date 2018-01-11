# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:26:39 2017

@author: abhishek
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

# load the boston dataset
boston = datasets.load_boston(return_X_y=False)
 
# defining feature matrix(X) and response vector(y)
X = boston.data
y = boston.target
 
# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)
## create correlation graph
#scatter_matrix(boston)
#plt.show()

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)
 
# regression coefficients
print('Coefficients: \n', reg.coef_)
 
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# RMSE for Training 
print ('For Training -----------------> ')
print("Mean squared error: %.2f"
      % mean_squared_error(y_train, reg.predict(X_train)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_train, reg.predict(X_train)))

# RMSE for test 
print ('For Test -----------------> ')
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, reg.predict(X_test)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, reg.predict(X_test)))


# plot for residual error
 
## setting plot style
plt.style.use('fivethirtyeight')
 
## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color = "red", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")
 
## function to show plot
plt.show()

## Plot Actual and predicted both the data set
plt.plot(y_test,label = 'Actual data')
plt.hlines(y = 40, xmin = 0, xmax = 250, linewidth = 2)
plt.plot(reg.predict(X_test),label = 'Predicted data')
plt.hlines(y = 10, xmin = 0, xmax = 250, linewidth = 2)
plt.legend(loc = 'upper right')
plt.show()