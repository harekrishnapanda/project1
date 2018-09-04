# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 15:52:50 2018

@author: Harekrishna
"""
#Import Libraries
import sqlite3
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Data Import.
cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

# To see the data co relation matrix
#df.corr(method = 'pearson')
#df.corr(method = 'kendall')
df.corr(method = 'spearman')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import numpy
df1 = df.iloc[:,5:42]
corelations =df1.corr()
fig=plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corelations, vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=numpytarange(0,37,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# Using user definded function for data cleansing
from data_preprocessing import data_cleaning
df= data_cleaning(df)

#Splitting data into X Axis and Y Axis
# Splitting X axis furhter based on categorical and numerical values for preprocessing
X1 = df.iloc[:,5]
X2 = df.iloc[:,9:]
X3 = df.iloc[:,[6,7,8]]
X4 = pd.concat([X1,X2],axis=1)
# creatin Y axis data
y = df['overall_rating']
#X3.describe(include='all')

# Label encoding of categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lblenc=LabelEncoder()

#sys.setrecursionlimit(200000)
X3['preferred_foot']=lblenc.fit_transform(X3['preferred_foot'].astype(str))
X3['defensive_work_rate']=lblenc.fit_transform(X3['defensive_work_rate'].astype(str))
X3['attacking_work_rate']=lblenc.fit_transform(X3['attacking_work_rate'].astype(str))
#X3.info()

# Merging of categorical and Numerical features
X = pd.concat([X3,X4],axis=1)

#Backword Elimination
import numpy as np
import statsmodels.formula.api as sma
X = np.append(arr = np.ones((176161,1)).astype(int),values = X, axis=1 )

regressor_sma = sma.OLS(endog=y, exog=X).fit()
regressor_sma.summary()

#Eliminating the columns based on Backward elimination result
X = pd.DataFrame(X)
X= X.drop([1],axis=1)

# Train_set and Test set data split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.90,random_state=0)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
lr.predict(X_test)
lrscore=lr.score(X_test,y_test)
lrscore
print('score with XGBoost is ', str(lrscore))
# Decission Tree Regression
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
y_pred= dtr.predict(X_test)
dtrscore=dtr.score(X_test,y_test)
dtrscore

#Random Forest Regression
# to get the optimum number of Trees
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict
ensemble_clfs = [ ("RandomForestRegressor, max_features=None",RandomForestRegressor(warm_start=True, max_features=None, oob_score=True,))]
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

min_estimators = 10
max_estimators = 300

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(X_train,y_train)
rf.predict(X_test)
fscore=rf.score(X_test,y_test)
print ('Score with Random Forest is ',str(fscore))

#fitting XGBoost to the Training set
import xgboost
xclassifier = xgboost.XGBRegressor()
xclassifier.fit(X_train,y_train)
xclassifier.predict(X_test)
xscore=xclassifier.score(X_test,y_test)
xscore
print('score with XGBoost is ',+ str(xscore))
