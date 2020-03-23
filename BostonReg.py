# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:45:33 2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Naming the column'''
names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MDEV']
dataset = pd.read_csv("C:/Users/user/Downloads/housing.csv",names = names,delim_whitespace= True)

'''Dimension of dataset'''
print(dataset.shape)

'''Check if any empty occurs'''
print(dataset.isnull().sum())

'''type of dataset'''
print(dataset.dtypes)

sns.distplot(dataset['MDEV'],bins = 30)
plt.show()

'''Correlation'''
correlation_mat = dataset.corr().round()
sns.heatmap(data = correlation_mat,annot =True)
plt.show()

'''spliting X and y'''
X = dataset.iloc[:,:13]
y = dataset.iloc[:,-1]

'''Spliting train and test dataset'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

'''Using LinearRegression'''
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train,y_train)

'''Predicting the result'''
y_pred = Regressor.predict(X_test)

'''Visualize the data'''
plt.scatter(y_test, y_pred, color='red')
plt.xlabel('Real Price', color='red')
plt.ylabel('Predicted Price', color='blue')
plt.plot(y_test, y_test + 1, '-o' , linestyle='solid',label='y=2x+1', color='blue')
plt.legend(loc='upper left')
plt.grid()
plt.show()

'''R and AdjR value'''
import statsmodels.api as sm
X = np.append(arr = np.ones((506, 1)).astype(int), values = X, axis = 1)
x_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

regressor_ols = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_ols.summary())

sm = Regressor.predict(X_test)
print(sm)

'''Mean Squared Error'''
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_pred)
print(MSE)

