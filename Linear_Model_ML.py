# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 22:03:19 2017

@author: abhishek

running ml on X (size) and Y (price)
"""
# Linear model
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# draw the corr plot
from pandas.tools.plotting import scatter_matrix

df = pd.read_csv("Housing.csv")
scatter_matrix(df)
plt.show()

Y = df['price']
X = df['lotsize']
 
X=X.reshape(len(X),1)
Y=Y.reshape(len(Y),1)
 
# Split the data into training/testing sets
X_train = X[:-250]
X_test = X[-250:]
 
# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]

# Plot outputs
plt.scatter(X_test, Y_test,  color='red')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
 
plt.show()

# Create linear reg object
regr = linear_model.LinearRegression()

# Train the model using the training data set
regr.fit(X_train,Y_train)
regr.score(X_train,Y_train)
print('coff', regr.coef_)
print('int', regr.intercept_)
# Plot the output 
plt.plot(X_test,regr.predict(X_test), color= 'red', linewidth=3)