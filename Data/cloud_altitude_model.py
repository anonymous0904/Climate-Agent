import random

import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import LSTM, Bidirectional, Dropout, Dense, Conv1D, MaxPooling1D
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from Data import csv_file_handler

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, input_cols, target_cols, sequence_length=10):
    scaler = MinMaxScaler()

    df_train = df[input_cols]
    df_train = df_train[:18671]
    df_train_scaled = scaler.fit_transform(df_train)
    df_test = df[['cloud_altitude']][18671:]

    air_temperature_pred = get_air_temperature_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_temperature'})
    dew_point_pred = get_dew_point_prediction_df().set_index('Time').rename(columns={'Train Prediction': 'dew_point'})
    air_pressure_pred = get_air_pressure_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_pressure'})

    df_test = df_test.join(
        [air_pressure_pred, dew_point_pred, air_temperature_pred])
    df_test = df_test[input_cols]
    df_test_scaled = scaler.transform(df_test)

    target_index = input_cols.index(target_cols)

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(df_train_scaled) - sequence_length):
        X_train.append(df_train_scaled[i:i + sequence_length])
        y_train.append(df_train_scaled[i + sequence_length, target_index])

    for i in range(len(df_test_scaled) - sequence_length):
        X_test.append(df_test_scaled[i:i + sequence_length])
        y_test.append(df_test_scaled[i + sequence_length, target_index])

    test_time = df_test.index[sequence_length:]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), test_time, scaler


# R² Score: 0.9256
def cloud_altitude_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def get_cloud_presence_prediction_df():
    cloud_presence_df = pd.read_csv('predictions/cloud_presence_prediction.csv')
    cloud_presence_df['Time'] = pd.to_datetime(cloud_presence_df['Time'])
    return cloud_presence_df[['Time', 'Train Prediction']]


def get_air_temperature_prediction_df():
    air_temperature_df = pd.read_csv('predictions/air_temperature_predictions.csv')
    air_temperature_df['Time'] = pd.to_datetime(air_temperature_df['Time'])
    return air_temperature_df[['Time', 'Train Prediction']]


def get_dew_point_prediction_df():
    dew_point_df = pd.read_csv('predictions/dew_point_predictions.csv')
    dew_point_df['Time'] = pd.to_datetime(dew_point_df['Time'])
    return dew_point_df[['Time', 'Train Prediction']]


def get_air_pressure_prediction_df():
    air_pressure_df = pd.read_csv('predictions/air_pressure_predictions.csv')
    air_pressure_df['Time'] = pd.to_datetime(air_pressure_df['Time'])
    return air_pressure_df[['Time', 'Train Prediction']]


def pad_predictions_for_inverse_transform(preds, total_features, target_index):
    padded = np.zeros((len(preds), total_features))
    padded[:, target_index] = preds.flatten()
    return padded


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(int)

metars_df_train = metars_df[:18671]
metars_df_test = metars_df[18671:]

metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
input_features = ['cloud_altitude', 'air_pressure', 'dew_point', 'air_temperature']
target_feature = 'cloud_altitude'

X_train, y_train, X_test, y_test, time_test, scaler = preprocess_data(metars_df, input_features, target_feature)

X_train_with_clouds = X_train[metars_df_train[10:]['cloud_presence'] == 1]
y_train_with_clouds = y_train[metars_df_train[10:]['cloud_presence'] == 1]
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

history = cloud_altitude_model.fit(X_train_with_clouds, y_train_with_clouds,
                                   validation_data=(X_test_with_clouds, y_test_with_clouds), epochs=10, batch_size=32)

cloud_altitude_predictions_with_clouds = cloud_altitude_model.predict(X_test_with_clouds)

target_index = input_features.index(target_feature)
padded_preds = pad_predictions_for_inverse_transform(cloud_altitude_predictions_with_clouds, len(input_features),
                                                     target_index)
padded_y_test = pad_predictions_for_inverse_transform(y_test, len(input_features), target_index)

cloud_altitude_predictions_with_clouds = scaler.inverse_transform(padded_preds)[:, target_index].flatten().astype(int)

n_test = len(get_cloud_presence_prediction_df())
y_test_unscaled = scaler.inverse_transform(padded_y_test)[:, target_index].flatten().astype(int)

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
