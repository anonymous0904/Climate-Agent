import random

import numpy as np
import pandas as pd
import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout
from sklearn.metrics import r2_score
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, input_features, target_feature, sequence_length=10):
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


# R² Score: 0.8762
def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def pad_predictions_for_inverse_transform(preds, total_features, target_index):
    padded = np.zeros((len(preds), total_features))
    padded[:, target_index] = preds.flatten()
    return padded


def predict_visibility_with_bilstm():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    input_features = ['predominant_horizontal_visibility', 'present_fog', 'precipitation', 'wind_direction',
                      'wind_speed', 'cloud_altitude', 'cloud_nebulosity', 'air_pressure', 'dew_point',
                      'air_temperature']
    target_feature = 'predominant_horizontal_visibility'

    X, y, scaler, observation_times = preprocess_data(metars_df, input_features, target_feature)

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

    bilstm_model = build_bilstm_model(X_train.shape[1:])
    bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    bilstm_predictions = bilstm_model.predict(X_test)

    target_index = input_features.index(target_feature)
    padded_preds = pad_predictions_for_inverse_transform(bilstm_predictions, len(input_features), target_index)
    padded_y_test = pad_predictions_for_inverse_transform(y_test, len(input_features), target_index)

    bilstm_predictions_unscaled = scaler.inverse_transform(padded_preds)[:, target_index].astype(int)
    y_test_unscaled = np.round(scaler.inverse_transform(padded_y_test)[:, target_index]).astype(int)

    return bilstm_predictions_unscaled, y_test_unscaled, time_test


visibility_predictions, visibility_test, time_test = predict_visibility_with_bilstm()
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

visibility_predictions = pd.Series(visibility_predictions).shift(-1)
visibility_predictions.iloc[-1] = 9999
print(f"R² Score: {r2_score(visibility_test, visibility_predictions):.4f}")

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': visibility_predictions.astype(int),
#           'Actual Value': visibility_test.flatten()})
#
# train_result.to_csv('predictions/visibility_predictions.csv', index=False, columns=train_result.columns)
