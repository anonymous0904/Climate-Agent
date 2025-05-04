import random

import numpy as np
import pandas as pd
from keras import Input
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt
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
    df = df[input_features]

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i + sequence_length][input_features].values)
        y.append(df.iloc[i + sequence_length][target_feature])

    observation_times = df.index[sequence_length:]
    return np.array(X), np.array(y), observation_times


# HYBRID 1D-CNN + BiLSTM - MODEL - ACCURACY: 0.9786
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


def predict_fog_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    input_features = ['present_fog', 'air_temperature', 'dew_point', 'air_pressure']
    target_feature = 'present_fog'

    scaler = MinMaxScaler()
    metars_df[input_features] = scaler.fit_transform(metars_df[input_features])

    X, y, observation_times = preprocess_data(metars_df, input_features, target_feature)
    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    time_train = observation_times[:split_index]
    time_test = observation_times[split_index:]

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
