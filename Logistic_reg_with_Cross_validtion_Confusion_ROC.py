# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:57:04 2017

@author: abhishek
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))

print(data.head())

# =============================================================================
# check for null values
# =============================================================================
data.isnull().sum()

# =============================================================================
# Check the independence between the independent variables
# =============================================================================
sns.heatmap(data.corr())
plt.show()

plt.savefig('pic/correlation')
# =============================================================================
# #Input variables
# #1 - age (numeric)
# #
# #2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# #
# #3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# #
# #4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# #
# #5 - default: has credit in default? (categorical: 'no','yes','unknown')
# #
# #6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# #
# #7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# #
# #8 - contact: contact communication type (categorical: 'cellular','telephone')
# #
# #9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# #
# #10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# #
# #11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# #
# #12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# #
# #13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# #
# #14 - previous: number of contacts performed before this campaign and for this client (numeric)
# #
# #15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# #
# #16 - emp.var.rate: employment variation rate - (numeric)
# #
# #17 - cons.price.idx: consumer price index - (numeric)
# #
# #18 - cons.conf.idx: consumer confidence index - (numeric)
# #
# #19 - euribor3m: euribor 3 month rate - (numeric)
# #
# #20 - nr.employed: number of employees - (numeric)
# #
# #Predict variable (desired target):
# #y - has the client subscribed a term deposit? (binary: '1','0')
# #
# #The education column of the dataset has many categories and we need to reduce the categories for a better modelling. The education column has the following categories
# =============================================================================

print(data['education'].unique())

 
# =============================================================================
# Let us group "basic.4y", "basic.9y" and "basic.6y" together and call them "basic".
# =============================================================================


data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])


print(data['education'].unique())

# =============================================================================
# Data exploration
# =============================================================================

print('class imbalace checks ')
print(data['y'].value_counts())

sns.countplot(x='y',data = data, palette = 'hls')
plt.show()
plt.savefig('pic/count_plot')

print(data.groupby('y').mean())
print(data.groupby('y').min())
print(data.groupby('y').max())

# =============================================================================
# Visualizations
# =============================================================================
#
##%matplotlib inline
pd.crosstab(data.job,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('pic/purchase_fre_job')
#
table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('pic/mariral_vs_pur_stack')
#
table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('pic/edu_vs_pur_stack')
#
pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pic/pur_dayofweek_bar')
#
pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pic/pur_fre_month_bar')
#
data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('pic/hist_age')
#
pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.savefig('pic/pur_fre_pout_bar')

# =============================================================================
# Create dummy variables
# =============================================================================

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
print(data_final.columns.values)
#
data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]


# =============================================================================
# Feature Selection
# =============================================================================

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y] )
print(rfe.support_)
print(rfe.ranking_)



cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
X=data_final[cols]
y=data_final['y']


# =============================================================================
# Implementing the model
# =============================================================================
# this is the error section
#import statsmodels.api as sm
#logit_model=sm.Logit(y,X)
#result=logit_model.fit()
#result.summary()

# =============================================================================
# Logistic Regression Model Fitting
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# =============================================================================
# Predicting the test set results and caculating the accuracy
# =============================================================================
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# =============================================================================
# Cross Validation
# =============================================================================

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

# =============================================================================
# Confusion Matrix
# =============================================================================

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# =============================================================================
# [[10872   109]
#  [ 1122   254]]
# The result is telling us that we have 10872+254 correct predictions and 1122+109 incorrect predictions.
# =============================================================================

# =============================================================================
# Compute precision, recall, F-measure and support
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
# 
# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
# 
# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
# 
# The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
# 
# The support is the number of occurrences of each class in y_test.
# =============================================================================
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# =============================================================================
# 
#              precision    recall  f1-score   support
# 
#           0       0.91      0.99      0.95     10981
#           1       0.70      0.18      0.29      1376
# 
# avg / total       0.88      0.90      0.87     12357
# 
# Interpretation:
# Of the entire test set, 88% of the promoted term deposit were the term deposit that the customers liked. Of the entire test set, 90% of the customer's preferred term deposit were promoted.
# =============================================================================
# =============================================================================
# ROC Curvefrom sklearn import metrics¶
# from ggplot import *
# 
# prob = clf1.predict_proba(Xtest)[:,1] fpr, sensitivity, = metrics.roc_curve(Y_test, prob)
# 
# df = pd.DataFrame(dict(fpr=fpr, sensitivity=sensitivity)) ggplot(df, aes(x='fpr', y='sensitivity')) +\ geom_line() +\ geom_abline(linetype='dashed')
# =============================================================================
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('pic/Log_ROC')
plt.show()