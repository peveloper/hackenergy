#!/usr/bin/env python
# coding: utf-8

# In[160]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from datetime import datetime
from pandas import concat
from pandas import Series
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

# get_ipython().run_line_magic('matplotlib', 'inline')



# In[142]:


consumption = pd.read_csv('data/production_consumption_2012_2016_scaled.csv', sep=';', decimal=',')
consumption['timestamp'] = pd.to_datetime(consumption['cet_cest_timestamp'])
consumption.drop(['utc_timestamp', 'cet_cest_timestamp', 'solarprod', 'windprod'], axis=1, inplace=True)
consumption.set_index('timestamp', inplace=True)
consumption = consumption[(consumption.index >= datetime(2012,1,1)) & (consumption.index < datetime(2016,12,1))]
consumption.head()


# In[143]:


w1 = pd.read_csv('data/weather_UTC_2012-2016_Berlin.csv', sep=';', decimal=',')
w1['timestamp'] = pd.to_datetime(w1['VALUE_TIME'])
w1.drop(['VALUE_TIME'], axis=1, inplace=True)
w1.set_index('timestamp', inplace=True)
w1 = w1[(w1.index >= datetime(2012,1,1)) & (w1.index < datetime(2016,12,1))]
w1.head()


# In[144]:


w2 = pd.read_csv('data/weather_UTC_2012-2016_Dusseldorf.csv', sep=';', decimal=',')
w2['timestamp'] = pd.to_datetime(w2['VALUE_TIME'])
w2.drop(['VALUE_TIME'], axis=1, inplace=True)
w2.set_index('timestamp', inplace=True)
w2 = w2[(w2.index >= datetime(2012,1,1)) & (w2.index < datetime(2016,12,1))]
w2.head()


# In[145]:


w3 = pd.read_csv('data/weather_UTC_2012-2016_Munich.csv', sep=';', decimal=',')
w3['timestamp'] = pd.to_datetime(w3['VALUE_TIME'])
w3.drop(['VALUE_TIME'], axis=1, inplace=True)
w3.set_index('timestamp', inplace=True)
w3 = w3[(w3.index >= datetime(2012,1,1)) & (w3.index < datetime(2016,12,1))]
w3.head()


# In[146]:


w = w1.merge(w2.merge(w3, how='outer', left_index=True, right_index=True), how='outer', left_index=True, right_index=True)


# In[147]:


w.head()


# In[148]:


X = w.merge(consumption, how='outer', left_index=True, right_index=True)


# In[149]:


X.head()


# In[150]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In[151]:


X.shape


# In[152]:


tfd = series_to_supervised(X.values, 1, 1, True)


# In[153]:


tfd


# In[155]:


def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled


# In[156]:


# split data into train and test-sets
train, test = tfd.values[0:-10000], tfd.values[-10000:]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)


# In[157]:


train_scaled.shape


# In[158]:


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# In[161]:


# repeat experiment
repeats = 30
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
	print('%d) Test RMSE: %.3f' % (r+1, rmse))
	error_scores.append(rmse)
 
# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
# results.boxplot()
# pyplot.show()


# In[ ]:




