import random

import numpy as np
import pandas as pd
from keras import Input

import csv_file_handler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, target_cols, sequence_length=10):
    df = df[target_cols]

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i + sequence_length].values)
        y.append(df.iloc[i + sequence_length].values)

    return np.array(X), np.array(y)


def build_wind_presence_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input shape is (sequence_length, num_features)
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

    X, y = preprocess_data(metars_df, ['wind_presence'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    wind_presence_model = build_wind_presence_model((X_train.shape[1], X_train.shape[2]))
    wind_presence_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    wind_presence_predictions = wind_presence_model.predict(X_test)
    return wind_presence_predictions, y_test


wind_presence_prediction, y_test = predict_wind_presence()
wind_presence_prediction_binary = (wind_presence_prediction > 0.5).astype(int)
wind_presence_test = y_test.astype(int)
print(f"Accuracy: {accuracy_score(wind_presence_test, wind_presence_prediction_binary):.4f}")

# train_result = pd.DataFrame(
#     data={'Train Prediction': wind_presence_prediction_binary.flatten(),
#           'Actual Value': wind_presence_test.flatten()})
# with open('predictions/wind_presence_prediction.txt', 'w') as f:
#     f.write(train_result.to_string())
