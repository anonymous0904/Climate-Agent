import random

import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import LSTM, Bidirectional, Dropout, Dense, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from keras.src.callbacks import EarlyStopping

from Data import csv_file_handler
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, input_features, target_feature, sequence_length=10):
    df = df[input_features]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df.iloc[i + sequence_length][target_feature])

    observation_times = df.index[sequence_length:]
    return np.array(X), np.array(y).astype(int), scaler, observation_times


def build_precipitation_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(int)
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
input_features = ['precipitation', 'cloud_presence', 'air_pressure', 'dew_point', 'air_temperature']
target_feature = 'precipitation'

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

precipitation_model = build_precipitation_model((X_train.shape[1], X_train.shape[2]))
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True, verbose=1)

precipitation_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32,
                        callbacks=[early_stopping])
precipitation_predictions = precipitation_model.predict(X_test)
precipitation_predictions = np.argmax(precipitation_predictions, axis=1)
precipitation_predictions = precipitation_predictions.astype(int)
y_test = y_test.astype(int)

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': precipitation_predictions.flatten(),
#           'Actual Value': y_test.flatten()})
#
# train_result.to_csv('predictions/precipitation_predictions.csv', index=False, columns=train_result.columns)
