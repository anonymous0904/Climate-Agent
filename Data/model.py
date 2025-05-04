import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Input, Dense, Dropout, LSTM, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

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


# input_shape = (sequence length, number of features)
# output_dim = number of predicted features
def build_bilstm_model(input_shape, output_dim):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# air_temperature, air_pressure, dew_point
def predict_with_bilstm(target_cols):
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    metars_df = metars_df.sort_index()
    X, y, scaler, observation_times = preprocess_data(metars_df, target_cols)

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

    bilstm_model = build_bilstm_model(X_train.shape[1:], len(target_cols))

    history = bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    bilstm_predictions = bilstm_model.predict(X_test)
    return scaler.inverse_transform(bilstm_predictions).astype(int), scaler.inverse_transform(y_test).astype(
        int), time_test, history

# target_columns = ['dew_point']
# bilstm_predictions, bilstm_test, time_test, history = predict_with_bilstm(
#     target_columns)  # 'air_temperature', 'dew_point', 'air_pressure'

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

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': bilstm_predictions.flatten(),
#           'Actual Value': bilstm_test.flatten()})
# train_result.sort_values(by=['Time'], ascending=True)
# train_result.to_csv('predictions/dew_point_predictions.csv', index=False, columns=train_result.columns)
