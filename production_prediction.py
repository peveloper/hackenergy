#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from datetime import datetime
from pandas import concat
from pandas import Series
from pandas import DataFrame

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    print('dio', train.shape)
    X, y = train[:, 0:-2], train[:, -2:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    print(X.shape)
    print(y.shape)
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

production = pd.read_csv('data/production_consumption_2012_2016_scaled.csv', sep=';', decimal=',')
production['timestamp'] = pd.to_datetime(production['cet_cest_timestamp'])
production.drop(['utc_timestamp', 'cet_cest_timestamp', 'consumption'], axis=1, inplace=True)
production.set_index('timestamp', inplace=True)
production = production[(production.index >= datetime(2012,1,1)) & (production.index < datetime(2016,12,1))]

w1 = pd.read_csv('data/weather_UTC_2012-2016_Berlin.csv', sep=';', decimal=',')
w1['timestamp'] = pd.to_datetime(w1['VALUE_TIME'])
w1.drop(['VALUE_TIME'], axis=1, inplace=True)
w1.set_index('timestamp', inplace=True)
w1 = w1[(w1.index >= datetime(2012,1,1)) & (w1.index < datetime(2016,12,1))]

w2 = pd.read_csv('data/weather_UTC_2012-2016_Dusseldorf.csv', sep=';', decimal=',')
w2['timestamp'] = pd.to_datetime(w2['VALUE_TIME'])
w2.drop(['VALUE_TIME'], axis=1, inplace=True)
w2.set_index('timestamp', inplace=True)
w2 = w2[(w2.index >= datetime(2012,1,1)) & (w2.index < datetime(2016,12,1))]

w3 = pd.read_csv('data/weather_UTC_2012-2016_Munich.csv', sep=';', decimal=',')
w3['timestamp'] = pd.to_datetime(w3['VALUE_TIME'])
w3.drop(['VALUE_TIME'], axis=1, inplace=True)
w3.set_index('timestamp', inplace=True)
w3 = w3[(w3.index >= datetime(2012,1,1)) & (w3.index < datetime(2016,12,1))]

w = w1.merge(w2.merge(w3, how='outer', left_index=True, right_index=True), how='outer', left_index=True, right_index=True)

X = w.merge(production, how='outer', left_index=True, right_index=True)

tfd = series_to_supervised(X.values, 1, 1, True)

# split data into train and test-sets
train, test = tfd.values[0:-10000], tfd.values[-10000:]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

print(train_scaled.shape)
print(test_scaled.shape)


# fit the model
lstm_model = fit_lstm(train_scaled, 1, 20, 4)
train_scaled = train_scaled[:,0:-2]
train_scaled = train_scaled.reshape(train_scaled.shape[0], 1, train_scaled.shape[1])
# Y_train, Y_pred = lstm_model.predict(train_scaled, batch_size=1, verbose=2)

# walk-forward validation on the test data
# predictions = list()
# ys = list()
# for i in range(len(test_scaled)):
    # make one-step forecast
    # X, y = test_scaled[i, 0:-2], test_scaled[i, -2]
    # yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    # yhat = invert_scale(scaler, X, yhat)
    # store forecast
    # predictions.append(yhat)
    # ys.append(y)


# report performance
# rmse = sqrt(mean_squared_error(ys, predictions))
# print('%d) Test RMSE: %.3f' % (1, rmse))


model_json = lstm_model.to_json()
with open("production_model.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
    lstm_model.save_weights("production_model.h5")
    print("Saved model to disk")
