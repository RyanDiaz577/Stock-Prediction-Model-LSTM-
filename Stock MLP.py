#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import pandas_datareader as web
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# In[4]:


#Get the stock quote

df = web.DataReader('VIZSF',data_source='yahoo', start='2021-01-01', end='2021-07-31')
print(df)


# In[5]:


# Vizualize the historical closing price in a plot

plt.figure(figsize=(16,8))
plt.title('Historical Closeing Prices')
plt.plot(df['Close'])
plt.xlabel('Date',Fontsize= 18)
plt.ylabel('Closing Price($)', fontsize= 18)
plt.show()


# In[10]:


# Create the dataframe that holds the machines training data
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
training_data_len


# In[9]:


# Scaling the data intp values between 0-1 for improved prediction accuracy 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[11]:


# Create the machines training data set (with scaled and reshaped values)
train_data = scaled_data[0:training_data_len,:]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()
  


# In[12]:


# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[13]:


# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
x_train.shape  


# In[14]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[15]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[16]:


# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[17]:


# Create the testing data set
test_data = scaled_data[training_data_len - 60:,:]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[18]:


# Convert the testing data to numpy array
x_test = np.array(x_test)


# In[19]:


# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))


# In[20]:


# Get the predictions (Unscale data to create usable information)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[21]:


# Root Mean Squared Error (testing the models accuracy)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[83]:


# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()


# 

# In[84]:


valid

