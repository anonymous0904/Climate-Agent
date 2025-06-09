import random

import numpy as np
import pandas as pd
from keras import Input
from keras.src.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import csv_file_handler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout, Conv1D, MaxPooling1D
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
    for i in range(len(df) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df.iloc[i + sequence_length][target_feature])

    observation_times = df.index[sequence_length:]
    return np.array(X), np.array(y), observation_times


def build_wind_presence_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def predict_wind_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df['wind_presence'] = (metars_df['wind_speed'] > 0).astype(int)
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    input_features = ['wind_presence', 'air_temperature', 'dew_point', 'air_pressure']
    target_feature = 'wind_presence'

    X, y, observation_times = preprocess_data(metars_df, input_features, target_feature)
    split_index = int(len(X) * 0.8)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    time_test = observation_times[split_index:]

    wind_presence_model = build_wind_presence_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True, verbose=1)
    history = wind_presence_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32,
                                      callbacks=[early_stopping])
    wind_presence_predictions = wind_presence_model.predict(X_test)
    return wind_presence_predictions, y_test, time_test, history


wind_presence_prediction, y_test, time_test, history = predict_wind_presence()
wind_presence_prediction_binary = (wind_presence_prediction > 0.5).astype(int)
wind_presence_test = y_test.astype(int)

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': wind_presence_prediction_binary.flatten(),
#           'Actual Value': wind_presence_test.flatten()})
#
# train_result.sort_values(by=['Time'], ascending=True)
# train_result.to_csv('predictions/wind_presence_prediction.csv', index=False, columns=train_result.columns)
