import random

import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt

import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential, Model
from keras.src.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Bidirectional, Reshape
from sklearn.model_selection import train_test_split
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def evaluate_angular_accuracy(true_angles, pred_angles, thresholds=None):
    if thresholds is None:
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
    df = df[input_features]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length, :-1])
        y.append(df_scaled[i + sequence_length, -2:])

    return np.array(X), np.array(y), scaler


def build_wind_direction_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def pad_predictions_for_inverse_transform(preds, total_features, target_index):
    padded = np.zeros((len(preds), total_features))
    padded[:, target_index] = preds.flatten()
    return padded


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
metars_df['hour'] = metars_df.index.hour
metars_df['month'] = metars_df.index.month
metars_df['wind_dir_sin'] = np.sin(np.deg2rad(metars_df['wind_direction']))
metars_df['wind_dir_cos'] = np.cos(np.deg2rad(metars_df['wind_direction']))
input_cols = ['air_temperature', 'dew_point', 'air_pressure', 'hour', 'month', 'wind_dir_sin', 'wind_dir_cos']
target_cols = ['wind_dir_sin', 'wind_dir_cos']
X, y, scaler = preprocess_data(metars_df, input_cols, target_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

model = build_wind_direction_model(X_train.shape[1:])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=[early_stopping])

predictions = model.predict(X_test)

pred_direction = (np.rad2deg(np.arctan2(predictions[:, 0], predictions[:, 1])) + 360) % 360
actual_direction = (np.rad2deg(np.arctan2(y_test[:, 0], y_test[:, 1])) + 360) % 360

angular_diff = np.abs(pred_direction - actual_direction)
angular_error = np.minimum(angular_diff, 360 - angular_diff)
mean_angular_error = np.mean(angular_error)

print(f"Mean Angular Error: {mean_angular_error:.2f}°")
angular_error = evaluate_angular_accuracy(actual_direction, pred_direction)

# train_result = pd.DataFrame(
#     data={'Train Prediction': pred_direction.astype(int),
#           'Actual Value': actual_direction.astype(int)})
# with open('predictions/wind_direction_prediction.txt', 'w') as f:
#     f.write(train_result.to_string())
