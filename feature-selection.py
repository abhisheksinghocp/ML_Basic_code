# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 22:39:59 2017

@author: abhishek
"""
# =============================================================================
# 1. Univariate Selection
# Statistical tests can be used to select those features that have the strongest relationship with the output variable.
# 
# The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.
# 
# The example below uses the chi squared (chi^2) statistical test for non-negative features to select 4 of the best features from the Pima Indians onset of diabetes dataset.
# 
# 
# =============================================================================
# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

#You can see the scores for each attribute and the 4 attributes chosen (those with the highest scores): plas, test, mass and age.

# =============================================================================
# 2. Recursive Feature Elimination
# The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.
# 
# It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.
# 
# You can learn more about the RFE class in the scikit-learn documentation.
# 
# The example below uses RFE with the logistic regression algorithm to select the top 3 features. The choice of algorithm does not matter too much as long as it is skillful and consistent.
# =============================================================================

print(' Feature Extraction with RFE')
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" %fit.n_features_)
print("Selected Features: %s" %fit.support_)
print("Feature Ranking: %s" %fit.ranking_)
#You can see that RFE chose the the top 3 features as preg, mass and pedi.
#
#These are marked True in the support_ array and marked with a choice “1” in the ranking_ array.

# =============================================================================
# 3. Principal Component Analysis
# Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.
# 
# Generally this is called a data reduction technique. A property of PCA is that you can choose the number of dimensions or principal component in the transformed result.
# 
# In the example below, we use PCA and select 3 principal components.
# 
# Learn more about the PCA class in scikit-learn by reviewing the PCA API. Dive deeper into the math behind PCA on the Principal Component Analysis Wikipedia article.
# =============================================================================
print('Feature Extraction with PCA')
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

# =============================================================================
# 
# 4. Feature Importance
# Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
# 
# In the example below we construct a ExtraTreesClassifier classifier for the Pima Indians onset of diabetes dataset. You can learn more about the ExtraTreesClassifier class in the scikit-learn API.
# =============================================================================
print(' Feature Importance with Extra Trees Classifier')
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

#You can see that we are given an importance score for each attribute where the larger score the more important the attribute. The scores suggest at the importance of plas, age and mass.