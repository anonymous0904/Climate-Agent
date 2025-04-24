import random

import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping

import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential, Model
from keras.src.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Bidirectional, Reshape
from sklearn.metrics import r2_score
import tensorflow as tf

# import matplotlib.pyplot as plt

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, input_features, target_feature, sequence_length=24):
    scaler = MinMaxScaler()

    df_train = df[input_features]
    df_train = df_train[:18671]
    df_train_scaled = scaler.fit_transform(df_train)
    df_test = df[['wind_speed']][18671:]

    wind_presence_pred = get_wind_presence_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'wind_presence'})
    wind_direction_pred = get_wind_direction_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'wind_direction'}
    )
    air_temperature_pred = get_air_temperature_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_temperature'})
    dew_point_pred = get_dew_point_prediction_df().set_index('Time').rename(columns={'Train Prediction': 'dew_point'})
    air_pressure_pred = get_air_pressure_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_pressure'})

    df_test = df_test.join(
        [wind_presence_pred, wind_direction_pred, air_temperature_pred, dew_point_pred, air_pressure_pred])

    df_test['hour'] = df['hour'][18671:]
    df_test['month'] = df['month'][18671:]

    df_test = df_test[input_features]
    df_test_scaled = scaler.transform(df_test)

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


# BiLSTM MODEL - R² Score: 0.9099
def build_wind_speed_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# 1D-CNN + BiLSTM MODEL - R² Score: 0.8948
# def build_wind_speed_model(input_shape):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Bidirectional(LSTM(64, return_sequences=True)))
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(64)))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1))
#
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     return model

def get_wind_presence_prediction_df():
    wind_presence_df = pd.read_csv('predictions/wind_presence_prediction.csv')
    wind_presence_df['Time'] = pd.to_datetime(wind_presence_df['Time'])
    return wind_presence_df[['Time', 'Train Prediction']]


def get_wind_direction_prediction_df():
    wind_direction_df = pd.read_csv('predictions/wind_direction_prediction.csv')
    wind_direction_df['Time'] = pd.to_datetime(wind_direction_df['Time'])
    return wind_direction_df[['Time', 'Train Prediction']]


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
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
metars_df['wind_speed'] = metars_df['wind_speed'].rolling(window=3, center=True).mean().bfill().ffill()
metars_df['hour'] = metars_df.index.hour
metars_df['month'] = metars_df.index.month
metars_df['wind_presence'] = (metars_df['wind_speed'] > 0).astype(int)
metars_df['wind_dir_sin'] = np.sin(np.deg2rad(metars_df['wind_direction']))
metars_df['wind_dir_cos'] = np.cos(np.deg2rad(metars_df['wind_direction']))
input_cols = ['wind_speed', 'wind_presence', 'wind_direction', 'air_temperature', 'dew_point',
              'air_pressure', 'hour',
              'month']
target_col = 'wind_speed'
X_train, y_train, X_test, y_test, time_test, scaler = preprocess_data(metars_df, input_cols, target_col)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

bilstm_model = build_wind_speed_model(X_train.shape[1:])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                           callbacks=[early_stopping])

bilstm_predictions = bilstm_model.predict(X_test)

target_index = input_cols.index(target_col)
padded_preds = pad_predictions_for_inverse_transform(bilstm_predictions, len(input_cols), target_index)
padded_y_test = pad_predictions_for_inverse_transform(y_test, len(input_cols), target_index)

bilstm_predictions_unscaled = scaler.inverse_transform(padded_preds)[:, target_index].astype(int)
y_test_unscaled = scaler.inverse_transform(padded_y_test)[:, target_index].astype(int)

wind_presence_prediction = get_wind_presence_prediction_df()[14:]['Train Prediction']
bilstm_predictions_unscaled = np.array(bilstm_predictions_unscaled) * np.array(wind_presence_prediction)

print(f"R² Score: {r2_score(bilstm_predictions_unscaled, y_test_unscaled):.4f}")

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': bilstm_predictions_unscaled.flatten(),
#           'Actual Value': y_test_unscaled.flatten()})
#
# train_result.to_csv('predictions/wind_speed_predictions.csv', index=False, columns=train_result.columns)

# plt.figure(figsize=(12, 5))
#
# # Loss
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Train Loss', color='blue')
# plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
# plt.title('Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()
#
# # MAE
# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='Train MAE', color='green')
# plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
# plt.title('MAE over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
