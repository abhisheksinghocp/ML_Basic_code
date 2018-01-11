# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:51:40 2017

@author: abhishek
"""

import statsmodels.formula.api as sm
import pandas as pd
import seaborn as sns

np.set_printoptions(suppress=True)

df = pd.read_csv("Housing.csv")
model1 = sm.ols(formula='price ~ lotsize + bedrooms + bathrms + stories + driveway + recroom + fullbase + gashw + airco + garagepl + prefarea', data=df)
fitted1 = model1.fit()
summary = fitted1.summary()
print(fitted1.summary())

# visualize the relationship between the features and the response using scatterplots
#sns.pairplot(data, x_vars=['TV','Radio','Newspaper'], y_vars='Sales', size=7, aspect=0.7)




#############################################

from scipy import stats
import numpy as np
x = np.random.random(10)
y = np.random.random(10)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

print(slope, intercept, r_value, p_value, std_err)