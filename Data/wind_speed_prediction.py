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

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, input_features, target_feature, sequence_length=24):
    df = df[input_features]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    target_index = input_features.index(target_feature)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length, :-1])
        y.append(df_scaled[i + sequence_length, target_index])

    observation_times = df.index[sequence_length:]
    return np.array(X), np.array(y), scaler, observation_times


# BiLSTM MODEL - R² Score: 0.9048
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


# 1D-CNN + BiLSTM MODEL - R² Score: 0.8672
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
def pad_predictions_for_inverse_transform(preds, total_features, target_index):
    padded = np.zeros((len(preds), total_features))
    padded[:, target_index] = preds.flatten()
    return padded


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
metars_df['wind_speed'] = metars_df['wind_speed'].rolling(window=3, center=True).mean().bfill().ffill()
metars_df['hour'] = metars_df.index.hour
metars_df['month'] = metars_df.index.month
metars_df['wind_dir_sin'] = np.sin(np.deg2rad(metars_df['wind_direction']))
metars_df['wind_dir_cos'] = np.cos(np.deg2rad(metars_df['wind_direction']))
input_cols = ['wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'air_temperature', 'dew_point', 'air_pressure', 'hour',
              'month']
target_col = 'wind_speed'
X, y, scaler, observation_times = preprocess_data(metars_df, input_cols, target_col)

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
time_train = observation_times[:split_index]
time_test = observation_times[split_index:]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

bilstm_model = build_wind_speed_model(X_train.shape[1:])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                 callbacks=[early_stopping])

bilstm_predictions = bilstm_model.predict(X_test)

target_index = input_cols.index(target_col)
padded_preds = pad_predictions_for_inverse_transform(bilstm_predictions, len(input_cols), target_index)
padded_y_test = pad_predictions_for_inverse_transform(y_test, len(input_cols), target_index)

bilstm_predictions_unscaled = scaler.inverse_transform(padded_preds)[:, target_index].astype(int)
y_test_unscaled = scaler.inverse_transform(padded_y_test)[:, target_index].astype(int)
print(f"R² Score: {r2_score(bilstm_predictions_unscaled, y_test_unscaled):.4f}")

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': bilstm_predictions_unscaled.flatten(),
#           'Actual Value': y_test_unscaled.flatten()})
#
# train_result.to_csv('predictions/wind_speed_predictions.csv', index=False, columns=train_result.columns)
