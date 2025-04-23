import random

import numpy as np
import pandas as pd
from keras import Input

import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout, Conv1D, MaxPooling1D
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, input_features, target_feature, sequence_length=10):
    scaler = MinMaxScaler()

    df_train = df[input_features]
    df_train = df_train[:18671]
    df_train_scaled = scaler.fit_transform(df_train)
    df_test = df[['predominant_horizontal_visibility']]
    df_test = df_test[18671:]

    fog_pred = get_fog_prediction_df().set_index('Time').rename(columns={'Train Prediction': 'present_fog'})
    precipitation_pred = get_precipitation_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'precipitation'})
    wind_direction_pred = get_wind_direction_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'wind_direction'})
    wind_speed_pred = get_wind_direction_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'wind_speed'})
    cloud_altitude_pred = get_cloud_altitude_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'cloud_altitude'})
    cloud_nebulosity_pred = get_cloud_nebulosity_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'cloud_nebulosity'})
    air_pressure_pred = get_air_pressure_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_pressure'})
    dew_point_pred = get_dew_point_prediction_df().set_index('Time').rename(columns={'Train Prediction': 'dew_point'})
    air_temperature_pred = get_air_temperature_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_temperature'})

    df_test = df_test.join(
        [fog_pred, precipitation_pred, wind_direction_pred, wind_speed_pred, cloud_altitude_pred, cloud_nebulosity_pred,
         air_pressure_pred, dew_point_pred, air_temperature_pred])
    df_test_scaled = scaler.fit_transform(df_test)
    target_index = input_features.index(target_feature)

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(df_train_scaled) - sequence_length):
        X_train.append(df_train_scaled[i:i + sequence_length])
        y_train.append(df_train_scaled[i + sequence_length, target_index])

    for i in range(len(df_test_scaled) - sequence_length):
        X_test.append(df_test_scaled[i:i + sequence_length])
        y_test.append(df_test_scaled[i + sequence_length, target_index])

    test_time = df_test.index[sequence_length:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), test_time, scaler


# BiLSTM: R² Score: 0.8762
# def build_visibility_model(input_shape):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(32)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1))
#
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

# CNN + BiLSTM: R² Score: 0.8865
def build_visibility_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def get_fog_prediction_df():
    fog_prediction_df = pd.read_csv('predictions/fog_predictions.csv')
    fog_prediction_df['Time'] = pd.to_datetime(fog_prediction_df['Time'])
    return fog_prediction_df[['Time', 'Train Prediction']]


def get_precipitation_prediction_df():
    precipitation_prediction_df = pd.read_csv('predictions/precipitation_predictions.csv')
    precipitation_prediction_df['Time'] = pd.to_datetime(precipitation_prediction_df['Time'])
    return precipitation_prediction_df[['Time', 'Train Prediction']]


def get_wind_direction_prediction_df():
    wind_direction_prediction_df = pd.read_csv('predictions/wind_direction_prediction.csv')
    wind_direction_prediction_df['Time'] = pd.to_datetime(wind_direction_prediction_df['Time'])
    return wind_direction_prediction_df[['Time', 'Train Prediction']]


def get_wind_speed_prediction_df():
    wind_speed_prediction_df = pd.read_csv('predictions/wind_speed_predictions.csv')
    wind_speed_prediction_df['Time'] = pd.to_datetime(wind_speed_prediction_df['Time'])
    return wind_speed_prediction_df[['Time', 'Train Prediction']]


def get_cloud_altitude_prediction_df():
    cloud_altitude_prediction_df = pd.read_csv('predictions/cloud_altitude_predictions.csv')
    cloud_altitude_prediction_df['Time'] = pd.to_datetime(cloud_altitude_prediction_df['Time'])
    return cloud_altitude_prediction_df[['Time', 'Train Prediction']]


def get_cloud_nebulosity_prediction_df():
    cloud_nebulosity_prediction_df = pd.read_csv('predictions/cloud_nebulosity_predictions.csv')
    cloud_nebulosity_prediction_df['Time'] = pd.to_datetime(cloud_nebulosity_prediction_df['Time'])
    return cloud_nebulosity_prediction_df[['Time', 'Train Prediction']]


def get_air_pressure_prediction_df():
    air_pressure_df = pd.read_csv('predictions/air_pressure_predictions.csv')
    air_pressure_df['Time'] = pd.to_datetime(air_pressure_df['Time'])
    return air_pressure_df[['Time', 'Train Prediction']]


def get_dew_point_prediction_df():
    dew_point_df = pd.read_csv('predictions/dew_point_predictions.csv')
    dew_point_df['Time'] = pd.to_datetime(dew_point_df['Time'])
    return dew_point_df[['Time', 'Train Prediction']]


def get_air_temperature_prediction_df():
    air_temperature_df = pd.read_csv('predictions/air_temperature_predictions.csv')
    air_temperature_df['Time'] = pd.to_datetime(air_temperature_df['Time'])
    return air_temperature_df[['Time', 'Train Prediction']]


def pad_predictions_for_inverse_transform(preds, total_features, target_index):
    padded = np.zeros((len(preds), total_features))
    padded[:, target_index] = preds.flatten()
    return padded


def predict_visibility():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    input_features = ['predominant_horizontal_visibility', 'present_fog', 'precipitation', 'wind_direction',
                      'wind_speed', 'cloud_altitude', 'cloud_nebulosity', 'air_pressure', 'dew_point',
                      'air_temperature']
    target_feature = 'predominant_horizontal_visibility'

    X_train, y_train, X_test, y_test, time_test, scaler = preprocess_data(metars_df, input_features, target_feature)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    visibility_model = build_visibility_model(X_train.shape[1:])
    visibility_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    visibility_predictions = visibility_model.predict(X_test)

    target_index = input_features.index(target_feature)
    padded_preds = pad_predictions_for_inverse_transform(visibility_predictions, len(input_features), target_index)
    padded_y_test = pad_predictions_for_inverse_transform(y_test, len(input_features), target_index)

    visibility_predictions_unscaled = scaler.inverse_transform(padded_preds)[:, target_index].astype(int)
    y_test_unscaled = np.round(scaler.inverse_transform(padded_y_test)[:, target_index]).astype(int)

    return visibility_predictions_unscaled, y_test_unscaled, time_test


visibility_predictions, visibility_test, time_test = predict_visibility()
for i in range(len(visibility_predictions)):
    if visibility_predictions[i] < 800:
        vis = visibility_predictions[i]
        visibility_predictions[i] = (vis // 50) * 50
    elif 800 < visibility_predictions[i] < 5000:
        vis = visibility_predictions[i]
        visibility_predictions[i] = (vis // 100) * 100
    elif 5000 < visibility_predictions[i] < 9999:
        vis = visibility_predictions[i]
        visibility_predictions[i] = (vis // 1000) * 1000
    elif 9999 <= visibility_predictions[i]:
        visibility_predictions[i] = 9999

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': visibility_predictions.astype(int),
#           'Actual Value': visibility_test.flatten()})
#
# train_result.to_csv('predictions/visibility_predictions.csv', index=False, columns=train_result.columns)
