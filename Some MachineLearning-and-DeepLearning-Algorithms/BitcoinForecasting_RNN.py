# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 08:50:59 2017

@author: shahik
"""

 
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

training_values=pd.read_csv('bitcoin_price_training.csv')
training_values=training_values[::-1] #reversing values order
training_values=training_values.iloc[:,1:2].values # takes column 1 as Vector/Array , 2 is excluded

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler() #Default feature range is b/w 0 and 1. 
training_values= sc.fit_transform(training_values)

 def create_timestepsdataset(look_back, dataset):
     dataX, dataY = [], []
     for i in range(look_back, len(dataset)):
                     dataX.append(dataset[i-look_back:i, 0])
                     dataY.append(dataset[i, 0])
     return np.array(dataX), np.array(dataY)
                  
look_back = 60
X_train, y_train = create_timestepsdataset(look_back,training_values)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) 


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM  # Type of RNN it has long memory 

regressor = Sequential() 

regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (look_back,1),return_sequences = True)) 
regressor.add(LSTM(units = 4, activation = 'tanh',input_shape=(look_back,1),return_sequences = True))

regressor.add(LSTM(units = 4, activation = 'sigmoid',input_shape=(look_back,1),return_sequences=True))
regressor.add(LSTM(units = 4, activation = 'tanh',input_shape=(look_back,1)))

regressor.add(Dense(units = 1)) 
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, batch_size = 32, epochs = 200) 

#regressor.summary()

# Getting the real stock price of 2017
test_values = pd.read_csv('bitcoin_priceTest.csv') 
real_stock_price = test_values.iloc[0:20000,3:4].values 

# Getting the predicted stock price 
inputs = real_stock_price
inputs = sc.transform(inputs) 
X_test, y_test = create_timestepsdataset(look_back,inputs)
# Reshaping: 2-D Array into 3-D Array from Keras docs 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # Getting original scale 

plt.plot(real_stock_price, color = 'red', label = 'Real Bitcoin Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Bitcoin Price')
plt.title('BitCoin Price Prediction')
plt.xlabel('TimeLine_Values')
plt.ylabel('BitCoin Price')
plt.legend()  
plt.show()

  
  
  