import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from datetime import datetime

energy_test = pd.read_csv('data/production_consumption_2017_scaled.csv', sep=';', decimal=',')
energy_test['timestamp'] = pd.to_datetime(energy_test['cet_cest_timestamp'])
energy_test.drop(['utc_timestamp', 'cet_cest_timestamp'], axis=1, inplace=True)
energy_test.set_index('timestamp', inplace=True)

energy_train = pd.read_csv('data/production_consumption_2012_2016_scaled.csv', sep=';', decimal=',')
energy_train['timestamp'] = pd.to_datetime(energy_train['cet_cest_timestamp'])
energy_train.drop(['utc_timestamp', 'cet_cest_timestamp'], axis=1, inplace=True)
energy_train.set_index('timestamp', inplace=True)
energy_train = energy_train[(energy_train.index >= datetime(2012,1,1)) & (energy_train.index < datetime(2016,12,31))]

weather_train = pd.read_csv('data/weather_UTC_2012-2016.csv', sep=';', decimal=',')
weather_train.VALUE_TIME = pd.to_datetime(weather_train.VALUE_TIME)
weather_train.set_index('VALUE_TIME', inplace=True)
weather_train = weather_train[(weather_train.index>=datetime(2012, 1, 1)) & (weather_train.index < datetime(2016,12,31))]

weather_test = pd.read_csv('data/weather_UTC_2017.csv', sep=';', decimal=',')
weather_test.VALUE_TIME = pd.to_datetime(weather_test.VALUE_TIME)
weather_test.set_index('VALUE_TIME', inplace=True)

# WIND POWER - LINEAR PREDICTION #
# Split the data into training/testing sets
x_train = np.array(weather_train[['wind_speed_10m','precip_1h', 't_2m']])
x_test = np.array(weather_test[['wind_speed_10m','precip_1h', 't_2m']][:24*365])

# Split the targets into training/testing sets
y_train = np.array(energy_train['windprod']).reshape(-1,1)
y_test = np.array(energy_test['windprod'][:24*365]).reshape(-1,1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Wind Prediction
wind_pred = regr.predict(x_test)
out = energy_test[['windprod']][:24*365].reset_index()
out['wind_prediction'] = wind_pred
out.to_csv('prediction/wind.csv', index=None)

# SOLAR POWER - LINEAR PREDICTION #
# Split the data into training/testing sets
x_train = np.array(weather_train['global_rad']).reshape(-1, 1)
x_test = np.array(weather_test['global_rad'][:24*365]).reshape(-1, 1)

# Split the targets into training/testing sets
y_train = np.array(energy_train['solarprod']).reshape(-1,1)
y_test = np.array(energy_test['solarprod'][:24*365]).reshape(-1,1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Wind Prediction
solar_pred = regr.predict(x_test)
out = energy_test[['solarprod']][:24*365].reset_index()
out['solar_prediction'] = solar_pred
out.to_csv('prediction/sun.csv', index=None)