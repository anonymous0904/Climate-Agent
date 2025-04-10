import random

import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import LSTM, Bidirectional, Dropout, Dense
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from Data import csv_file_handler

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, target_cols, sequence_length=10):
    df = df[target_cols]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length, [df.columns.get_loc(col) for col in target_cols]])

    return np.array(X), np.array(y), scaler


# BiLSTM-Model
def cloud_altitude_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(int)
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
input_features = ['cloud_altitude', 'cloud_presence']
target_feature = ['cloud_altitude']

X, y, scaler = preprocess_data(metars_df, target_feature)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

cloud_altitude_model = cloud_altitude_model((X_train.shape[1], X_train.shape[2]))

cloud_altitude_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

cloud_altitude_predictions = cloud_altitude_model.predict(X_test)
cloud_altitude_predictions = scaler.inverse_transform(cloud_altitude_predictions).astype(int)
y_test = scaler.inverse_transform(y_test).astype(int)

print(f"R² Score: {r2_score(y_test, cloud_altitude_predictions):.4f}")
# R² Score: 0.5312

# train_result = pd.DataFrame(
#     data={'Train Prediction': bilstm_predictions.flatten(),
#           'Actual Value': y_test.flatten()})
# with open('predictions/cloud_altitude_predictions.txt', 'w') as f:
#     f.write(train_result.to_string())
