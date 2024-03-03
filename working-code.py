#!/usr/bin/env python
# coding: utf-8

# ### Stock Market Prediction And Forecasting Using Stacked LSTM

# In[170]:


### Keras and Tensorflow >2.0


# In[267]:


### Data Collection
import pandas_datareader as pdr
#api_key="bedfdf679238524b4fdd2779f5fe8ca942202df7"


# In[268]:


df = pdr.get_data_tiingo('TSLA', api_key = "bedfdf679238524b4fdd2779f5fe8ca942202df7")


# In[269]:


df. to_csv('TSLA.csv')


# In[271]:


df.head


# In[272]:


import pandas as pd


# In[273]:


df=pd.read_csv('TSLA.csv')


# In[ ]:





# In[274]:


df1=df.reset_index()['close']


# In[275]:


df1


# In[276]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[277]:


### LSTM are sensitive to the scale of the data. so we apply MinMax scaler 


# In[278]:


import numpy as np


# In[279]:


df1


# In[280]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[281]:


print(df1)


# In[282]:


##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[283]:


training_size,test_size


# In[284]:


#train_data


# In[285]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[226]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[228]:


print(X_train.shape), print(y_train.shape)


# In[230]:


print(X_test.shape), print(ytest.shape)


# In[232]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[234]:


### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[236]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[238]:


model.summary()


# In[240]:


model.summary()


# In[ ]:





# In[242]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# In[244]:


import tensorflow as tf


# In[246]:


tf.__version__


# In[248]:


### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[249]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[250]:


### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[251]:


### Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[252]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[253]:


len(test_data)


# In[254]:


x_input=test_data[341:].reshape(1,-1)
x_input.shape


# In[ ]:





# In[ ]:





# In[255]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[256]:


temp_input


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




