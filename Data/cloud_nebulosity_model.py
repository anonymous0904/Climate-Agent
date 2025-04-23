import random

import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import LSTM, Bidirectional, Dropout, Dense, Conv1D, MaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from Data import csv_file_handler
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
    df_train_scaled = scaler.fit_transform(df_train)
    df_test = df[['cloud_nebulosity']]
    df_test = df_test[18671:]
    test_index = df_test.index

    cloud_presence_pred = get_cloud_presence_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'cloud_presence'})
    air_pressure_pred = get_air_pressure_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_pressure'})
    air_temperature_pred = get_air_temperature_prediction_df().set_index('Time').rename(
        columns={'Train Prediction': 'air_temperature'})
    dew_point_pred = get_dew_point_prediction_df().set_index('Time').rename(columns={'Train Prediction': 'dew_point'})

    df_test = df_test.join([cloud_presence_pred, air_pressure_pred, dew_point_pred, air_temperature_pred])
    df_test_scaled = scaler.fit_transform(df_test)

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(df_train) - sequence_length):
        X_train.append(df_train_scaled[i:i + sequence_length])
        y_train.append(df_train.iloc[i + sequence_length][target_feature])

    for i in range(len(df_test) - sequence_length):
        X_test.append(df_test_scaled[i:i + sequence_length])
        y_test.append(df_test.iloc[i + sequence_length][target_feature])

    test_time = df_test.index[sequence_length:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), test_time


# BiLSTM - Accuracy: 0.8280
# def build_cloud_nebulosity_model(input_shape):
#     model = Sequential()
#     model.add(Input(shape=input_shape))  # Input shape will be (sequence_length, num_features)
#     model.add(Bidirectional(LSTM(64, return_sequences=True)))
#     model.add(Dropout(0.2))
#     model.add(Bidirectional(LSTM(32)))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(5, activation='softmax'))
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model


# CNN + BiLSTM - Accuracy: 0.8995
def build_cloud_nebulosity_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_cloud_presence_prediction_df():
    cloud_presence_df = pd.read_csv('predictions/cloud_presence_prediction.csv')
    cloud_presence_df['Time'] = pd.to_datetime(cloud_presence_df['Time'])
    return cloud_presence_df[['Time', 'Train Prediction']]


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


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(int)
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
input_features = ['cloud_nebulosity', 'cloud_presence', 'air_pressure', 'dew_point', 'air_temperature']
target_feature = 'cloud_nebulosity'

X_train, y_train, X_test, y_test, time_test = preprocess_data(metars_df, input_features, target_feature)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

cloud_nebulosity_model = build_cloud_nebulosity_model((X_train.shape[1], X_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True, verbose=1)

cloud_nebulosity_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32,
                           callbacks=[early_stopping])
cloud_nebulosity_predictions = cloud_nebulosity_model.predict(X_test)
cloud_nebulosity_predictions = np.argmax(cloud_nebulosity_predictions, axis=1)
cloud_nebulosity_predictions = pd.Series(cloud_nebulosity_predictions)
cloud_nebulosity_predictions = pd.Series(cloud_nebulosity_predictions).shift(-1)
cloud_nebulosity_predictions.iloc[-1] = 0
cloud_nebulosity_predictions = cloud_nebulosity_predictions.astype(int)
y_test = y_test.astype(int)
print(f"Accuracy: {accuracy_score(cloud_nebulosity_predictions, y_test):.4f}")

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': cloud_nebulosity_predictions,
#           'Actual Value': y_test.flatten()})
# train_result.to_csv('predictions/cloud_nebulosity_predictions.csv', index=False, columns=train_result.columns)
