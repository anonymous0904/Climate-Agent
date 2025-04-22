import random

import numpy as np
import pandas as pd
from keras import Input
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import csv_file_handler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, input_features, target_feature, sequence_length=10):
    scaler = MinMaxScaler()

    df_train = df[input_features]
    df_train = df_train[:18671]
    train_index = df_train.index
    df_train = scaler.fit_transform(df_train)

    air_pressure_pred = get_air_pressure_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_pressure'})
    air_temperature_pred = get_air_temperature_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_temperature'})
    dew_point_pred = get_dew_point_prediction_df().set_index('Time').rename(columns={'Train Prediction': 'dew_point'})

    df_test = df[['present_fog']]
    df_test = df_test[18671:]
    test_index = df_test.index

    df_test = df_test.join([air_temperature_pred, dew_point_pred, air_pressure_pred])
    df_test = scaler.fit_transform(df_test)

    df_train = pd.DataFrame(df_train, columns=input_features, index=train_index)
    df_test = pd.DataFrame(df_test, columns=input_features, index=test_index)

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(df_train) - sequence_length):
        X_train.append(df_train.iloc[i:i + sequence_length][input_features].values)
        y_train.append(df_train.iloc[i + sequence_length][target_feature])

    for i in range(len(df_test) - sequence_length):
        X_test.append(df_test.iloc[i:i + sequence_length][input_features].values)
        y_test.append(df_test.iloc[i + sequence_length][target_feature])

    test_time = df_test.index[sequence_length:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), test_time


# BINARY CLASSIFICATION MODELS FOR THE PRESENCE OF FOG

# BiLSTM - MODEL - ACCURACY: 0.9897
# def build_fog_presence_model(input_shape):
#     model = Sequential()
#     model.add(Input(shape=input_shape))  # Input shape will be (sequence_length, num_features)
#     model.add(Bidirectional(LSTM(64, return_sequences=True)))
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(32)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# HYBRID 1D-CNN + BiLSTM - MODEL - ACCURACY: 0.9886
def build_fog_presence_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


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


def predict_fog_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    input_features = ['present_fog', 'air_temperature', 'dew_point', 'air_pressure']
    target_feature = 'present_fog'

    X_train, y_train, X_test, y_test, time_test = preprocess_data(metars_df, input_features, target_feature)

    fog_presence_model = build_fog_presence_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True, verbose=1)
    fog_presence_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32,
                           callbacks=[early_stopping])
    fog_presence_predictions = fog_presence_model.predict(X_test)
    return fog_presence_predictions, y_test, time_test


fog_presence_prediction, fog_presence_test, time_test = predict_fog_presence()
fog_presence_prediction_binary = (fog_presence_prediction > 0.5).astype(int)
fog_presence_test = fog_presence_test.astype(int)
print(f"Accuracy: {accuracy_score(fog_presence_test, fog_presence_prediction_binary):.4f}")

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': fog_presence_prediction_binary.flatten(),
#           'Actual Value': fog_presence_test.flatten()})
# train_result.to_csv('predictions/fog_predictions.csv', index=False, columns=train_result.columns)
