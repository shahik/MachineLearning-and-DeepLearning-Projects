# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:45:33 2017

@author: shahik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from xgboost import XGBClassifier 

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')



X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 4].values
              
  #Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X=X[:,1:] #removed 1st column
              
 #Split in training and test set             
from sklearn.cross_validation import train_test_split
X_train,X_test, Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the Training set 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results
y_pred=regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#adding x0 column of 1 to tell statsmodels it exits
X= np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
 
X_opt=X[:,[0,1,2,3,4,5]]
#step 2 from notes
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()
#step 3 P>SL or 0.05 remove predictor 
X_opt=X[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()


X_opt=X[:,[0,3]]
regressor_OLS=sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

"""X_train_opt=X_opt[:40,-1]

#Visualising the Training set results
plt.scatter(X_train_opt,Y_train,color='red')
plt.plot(X_train_opt,regressor.predict(X_train_opt),color='blue')
plt.title("Salary vs Experience(Training set)")
plt.xlabel('Research and Development')
plt.ylabel('Profit')
plt.show()

#Visualising the Test set results
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience(Test set)")
plt.xlabel('Research and Development')
plt.ylabel('Profit')
plt.show()  """











