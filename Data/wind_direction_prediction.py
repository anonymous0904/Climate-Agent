import random

import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping

import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential, Model
from keras.src.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Bidirectional, Reshape
import tensorflow as tf
import matplotlib.pyplot as plt

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def evaluate_angular_accuracy(true_angles, pred_angles):
    thresholds = [5, 10, 20, 30, 45, 60]
    angular_diff = np.abs(pred_angles - true_angles)
    angular_error = np.minimum(angular_diff, 360 - angular_diff)

    print(f"\nAngular Accuracy Report (based on {len(true_angles)} samples):")
    for threshold in thresholds:
        pct = np.mean(angular_error <= threshold) * 100
        print(f"  - Within ±{threshold}°: {pct:.2f}%")

    mean_err = np.mean(angular_error)
    median_err = np.median(angular_error)
    max_err = np.max(angular_error)

    print(f"\nError values:")
    print(f"  - Mean Angular Error: {mean_err:.2f}°")
    print(f"  - Median Angular Error: {median_err:.2f}°")
    print(f"  - Max Angular Error: {max_err:.2f}°")

    return angular_error


def preprocess_data(df, input_features, sequence_length=24):
    scaler = MinMaxScaler()

    df_train = df[input_features]
    df_train = df_train[:18671]
    df_train_scaled = scaler.fit_transform(df_train)
    df_test = df[['hour', 'month']]
    df_test = df_test[18671:]

    wind_presence_pred = get_wind_presence_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'wind_presence'})
    air_temperature_pred = get_air_temperature_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_temperature'})
    dew_point_pred = get_dew_point_prediction_df().set_index('Time').rename(columns={'Train Prediction': 'dew_point'})
    air_pressure_pred = get_air_pressure_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_pressure'})

    df_test = df_test.join([wind_presence_pred, air_temperature_pred, dew_point_pred, air_pressure_pred])
    df_test['wind_dir_sin'] = df['wind_dir_sin'][18671:]
    df_test['wind_dir_cos'] = df['wind_dir_cos'][18671:]
    df_test = df_test[input_features]
    df_test_scaled = scaler.fit_transform(df_test)

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(df_train_scaled) - sequence_length):
        X_train.append(df_train_scaled[i:i + sequence_length])
        y_train.append(df_train_scaled[i + sequence_length, -2:])

    for i in range(len(df_test_scaled) - sequence_length):
        X_test.append(df_test_scaled[i:i + sequence_length])
        y_test.append(df_test_scaled[i + sequence_length, -2:])

    test_time = df_test.index[sequence_length:]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), test_time, scaler


# CNN + BiLSTM - Mean Angular Error: 10.58°
def build_wind_direction_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def get_wind_presence_prediction_df():
    wind_presence_df = pd.read_csv('predictions/wind_presence_prediction.csv')
    wind_presence_df['Time'] = pd.to_datetime(wind_presence_df['Time'])
    return wind_presence_df[['Time', 'Train Prediction']]


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


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
metars_df['hour'] = metars_df.index.hour
metars_df['month'] = metars_df.index.month
metars_df['wind_presence'] = (metars_df['wind_speed'] > 0).astype(int)
metars_df['wind_direction'] = metars_df['wind_direction'].rolling(window=3, center=True).mean().bfill().ffill()
metars_df['wind_dir_sin'] = np.sin(np.deg2rad(metars_df['wind_direction']))
metars_df['wind_dir_cos'] = np.cos(np.deg2rad(metars_df['wind_direction']))
input_cols = ['hour', 'month', 'wind_presence', 'air_temperature', 'dew_point', 'air_pressure', 'wind_dir_sin',
              'wind_dir_cos']
target_cols = ['wind_dir_sin', 'wind_dir_cos']

X_train, y_train, X_test, y_test, time_test, scaler = preprocess_data(metars_df, input_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

model = build_wind_direction_model(X_train.shape[1:])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                    callbacks=[early_stopping])

predictions = model.predict(X_test)

pred_direction = (np.rad2deg(np.arctan2(predictions[:, 0], predictions[:, 1])) + 361) % 361
actual_direction = (np.rad2deg(np.arctan2(y_test[:, 0], y_test[:, 1])) + 361) % 361

angular_diff = np.abs(pred_direction - actual_direction)
angular_error = np.minimum(angular_diff, 360 - angular_diff)
mean_angular_error = np.mean(angular_error)

wind_presence_prediction = get_wind_presence_prediction_df()[14:]['Train Prediction']
pred_direction = np.array(pred_direction) * np.array(wind_presence_prediction)

print(f"Mean Angular Error: {mean_angular_error:.2f}°")
angular_error = evaluate_angular_accuracy(actual_direction, pred_direction)

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': pred_direction.astype(int),
#           'Actual Value': actual_direction.astype(int)})
#
# train_result.to_csv('predictions/wind_direction_prediction.csv', index=False, columns=train_result.columns)

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
