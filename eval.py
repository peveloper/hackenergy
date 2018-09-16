from keras.models import model_from_json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from pandas import concat

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
lstm_model = model_from_json(loaded_model_json)
# load weights into new model
lstm_model.load_weights("model/model.h5")
print("Loaded model from disk")

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


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

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


consumption = pd.read_csv('data/production_consumption_2012_2016_scaled.csv', sep=';', decimal=',')
consumption['timestamp'] = pd.to_datetime(consumption['cet_cest_timestamp'])
consumption.drop(['utc_timestamp', 'cet_cest_timestamp', 'solarprod', 'windprod'], axis=1, inplace=True)
consumption.set_index('timestamp', inplace=True)
consumption = consumption[(consumption.index >= datetime(2012,1,1)) & (consumption.index < datetime(2016,12,1))]

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

X = w.merge(consumption, how='outer', left_index=True, right_index=True)

tfd = series_to_supervised(X.values, 1, 1, True)

print(tfd.shape)


# split data into train and test-sets
train, test = tfd.values[0:-10000], tfd.values[-10000:]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

print(train_scaled.shape)
print(test_scaled.shape)


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# X = tfd.values
# X = X[:,:-1]
# print(X.shape)
# Y = tfd.values
# Y = Y[:,-1:]
# print(Y.shape)

# X = X.reshape(X.shape[0], 1, X.shape[1])
# Y = Y.reshape(Y.shape[0], Y.shape[1])


# print(X.shape)
# print(Y.shape)

lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# walk-forward validation on the test data
predictions = list()
ys = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    y = invert_scale(scaler, X, y)
    # store forecast
    predictions.append(yhat)
    ys.append(y)

with open('pred.txt', 'w') as file:
        file.write(str(ys))
        file.write('\n')
        file.write(str(predictions))


