import random

import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import LSTM, Bidirectional, Dropout, Dense, Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from Data import csv_file_handler

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, target_cols, sequence_length=10):
    df = df[target_cols]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length, [df.columns.get_loc(col) for col in target_cols]])

    observation_times = df.index[sequence_length:]
    return np.array(X), np.array(y), scaler, observation_times


# BiLSTM-Model - R² Score: 0.8738
def cloud_altitude_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# R² Score: 0.8641
# def cloud_altitude_model(input_shape):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.3))
#     model.add(Bidirectional(LSTM(64, return_sequences=True)))
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(32)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1))
#
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model


def get_cloud_presence_prediction_df():
    cloud_presence_df = pd.read_csv('predictions/cloud_presence_prediction.csv')
    cloud_presence_df['Time'] = pd.to_datetime(cloud_presence_df['Time'])
    return cloud_presence_df[['Time', 'Train Prediction']]


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(int)
metars_df['cloud_altitude_mean3'] = metars_df['cloud_altitude'].rolling(3).mean()
metars_df['cloud_altitude'] = metars_df['cloud_altitude'].clip(lower=0, upper=200)

metars_df_train = metars_df[:18671]
metars_df_test = metars_df[18671:]

metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
input_features = ['cloud_altitude', 'cloud_altitude_mean3', 'cloud_presence', 'air_pressure',
                  'dew_point',
                  'air_temperature']
target_feature = ['cloud_altitude']

X, y, scaler, observation_times = preprocess_data(metars_df, target_feature)
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
time_train = pd.to_datetime(observation_times[:split_index])
time_test = pd.to_datetime(observation_times[split_index:])

X_train_with_clouds = X_train[metars_df_train['cloud_presence'] == 1]
y_train_with_clouds = y_train[metars_df_train['cloud_presence'] == 1]
X_test_with_clouds = X_test[get_cloud_presence_prediction_df()['Train Prediction'] == 1]
y_test_with_clouds = y_test[get_cloud_presence_prediction_df()['Train Prediction'] == 1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
X_train_with_clouds = X_train_with_clouds.astype('float32')
y_train_with_clouds = y_train_with_clouds.astype('float32')
X_test_with_clouds = X_test_with_clouds.astype('float32')
y_test_with_clouds = y_test_with_clouds.astype('float32')

cloud_altitude_model = cloud_altitude_model((X_train.shape[1], X_train.shape[2]))

cloud_altitude_model.fit(X_train_with_clouds, y_train_with_clouds,
                         validation_data=(X_test_with_clouds, y_test_with_clouds), epochs=30, batch_size=32)

cloud_altitude_predictions_with_clouds = cloud_altitude_model.predict(X_test_with_clouds)
cloud_altitude_predictions_with_clouds = scaler.inverse_transform(
    cloud_altitude_predictions_with_clouds).flatten().astype(int)
y_test_with_clouds = scaler.inverse_transform(y_test_with_clouds).astype(int)

n_test = len(get_cloud_presence_prediction_df())
y_test_unscaled = scaler.inverse_transform(y_test).flatten().astype(int)

altitude_preds = np.zeros(n_test)
altitude_preds[get_cloud_presence_prediction_df()['Train Prediction'] == 1] = cloud_altitude_predictions_with_clouds
altitude_preds[get_cloud_presence_prediction_df()['Train Prediction'] == 0] = 0
altitude_preds = pd.Series(altitude_preds).shift(-1)
altitude_preds.iloc[-1] = 0
altitude_preds = altitude_preds.astype(int)

print(f"R² Score: {r2_score(y_test_unscaled, altitude_preds):.4f}")

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': altitude_preds,
#           'Actual Value': y_test_unscaled})
#
# train_result.to_csv('predictions/cloud_altitude_predictions.csv', index=False, columns=train_result.columns)
