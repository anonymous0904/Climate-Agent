import numpy as np
import pandas as pd
from keras import Input

import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preprocess_cloud_presence_data(df, input_features, target_feature, sequence_length=10):
    df = df[input_features]

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i + sequence_length][input_features].values)
        y.append(df.iloc[i + sequence_length][target_feature])

    return np.array(X), np.array(y)


# BINARY CLASSIFICATION MODELS FOR THE PRESENCE OF CLOUDS

# BiLSTM - MODEL  - Accuracy: 0.8972
# def build_cloud_presence_model(input_shape):
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


# 1D-CNN - MODEL  - Accuracy: 0.8944
# def build_cloud_presence_model(input_shape):
#     model = Sequential()
#     model.add(Input(shape=input_shape))
#     model.add(Conv1D(64, kernel_size=3, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.3))
#     model.add(Flatten())
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model


def predict_cloud_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(int)
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    input_features = ['cloud_presence', 'air_temperature', 'dew_point', 'air_pressure']
    target_feature = 'cloud_presence'

    scaler = MinMaxScaler()
    metars_df[input_features] = scaler.fit_transform(metars_df[input_features])

    X, y = preprocess_cloud_presence_data(metars_df, input_features, target_feature)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cloud_presence_model = build_cloud_presence_model((X_train.shape[1], X_train.shape[2]))
    cloud_presence_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    cloud_presence_predictions = cloud_presence_model.predict(X_test)
    return cloud_presence_predictions, y_test


cloud_presence_prediction, cloud_presence_test = predict_cloud_presence()
cloud_presence_prediction_binary = (cloud_presence_prediction > 0.5).astype(int)
cloud_presence_test = cloud_presence_test.astype(int)
print(f"Accuracy: {accuracy_score(cloud_presence_test, cloud_presence_prediction_binary):.4f}")

# train_result = pd.DataFrame(
#     data={'Train Prediction': cloud_presence_prediction_binary.flatten(),
#           'Actual Value': cloud_presence_test.flatten()})
# with open('predictions/cloud_presence_prediction.txt', 'w') as f:
#     f.write(train_result.to_string())
