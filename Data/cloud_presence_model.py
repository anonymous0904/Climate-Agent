import random

import numpy as np
import pandas as pd
from keras import Input
from keras.src.callbacks import EarlyStopping

import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_cloud_presence_data(df, input_features, target_feature, sequence_length=10):
    df = df[input_features]

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i + sequence_length][input_features].values)
        y.append(df.iloc[i + sequence_length][target_feature])

    observation_times = df.index[sequence_length:]
    return np.array(X), np.array(y), observation_times


# HYBRID 1D-CNN + BiLSTM - MODEL - Accuracy: 0.8972
def build_cloud_presence_model(input_shape):
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


def predict_cloud_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(int)
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    metars_df['hour'] = metars_df.index.hour
    metars_df['month'] = metars_df.index.month
    input_features = ['cloud_presence', 'air_temperature', 'dew_point', 'air_pressure', 'hour', 'month']
    target_feature = 'cloud_presence'

    scaler = MinMaxScaler()
    metars_df[input_features] = scaler.fit_transform(metars_df[input_features])

    X, y, observation_times = preprocess_cloud_presence_data(metars_df, input_features, target_feature)
    split_index = int(len(X) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    time_train = observation_times[:split_index]
    time_test = observation_times[split_index:]

    cloud_presence_model = build_cloud_presence_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True, verbose=1)
    cloud_presence_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32,
                             callbacks=[early_stopping])
    cloud_presence_predictions = cloud_presence_model.predict(X_test)
    return cloud_presence_predictions, y_test, time_test


cloud_presence_prediction, cloud_presence_test, time_test = predict_cloud_presence()
cloud_presence_prediction_binary = (cloud_presence_prediction > 0.5).astype(int)
cloud_presence_test = cloud_presence_test.astype(int)
print(f"Accuracy: {accuracy_score(cloud_presence_test, cloud_presence_prediction_binary):.4f}")

# train_result = pd.DataFrame(
#     data={'Time': time_test,
#           'Train Prediction': cloud_presence_prediction_binary.flatten(),
#           'Actual Value': cloud_presence_test.flatten()})
# train_result.sort_values(by=['Time'], ascending=True)
# train_result.to_csv('predictions/cloud_presence_prediction.csv', index=False, columns=train_result.columns)
