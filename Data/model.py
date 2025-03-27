import numpy as np
import pandas as pd
import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential, Model
from keras.src.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Bidirectional, Reshape
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def preprocess_data(df, target_cols, sequence_length=10):
    df = df.drop(columns=['observation_time'])
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length, [df.columns.get_loc(col) for col in target_cols]])

    return np.array(X), np.array(y), scaler


# input_shape = (sequence length, number of features)
# output_dim = number of predicted features
def build_cnn_model(input_shape, output_dim):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# input_shape = (sequence length, number of features)
# output_dim = number of predicted features
def build_bilstm_model(input_shape, output_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_dim))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def predict_with_cnn(target_cols):
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    X, y, scaler = preprocess_data(metars_df, target_cols)
    # split data into train and test information
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # build the models
    cnn_model = build_cnn_model(X_train.shape[1:], len(target_cols))
    # train the models
    cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)
    # calculate predictions for variables using the test data
    cnn_predictions = cnn_model.predict(X_test)
    return cnn_predictions, y_test


# air_temperature, air_pressure, dew_point
def predict_with_bilstm(target_cols):
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    X, y, scaler = preprocess_data(metars_df, target_cols)
    # split data into train and test information
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # build the models
    bilstm_model = build_bilstm_model(X_train.shape[1:], len(target_cols))

    # train the models
    bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    # calculate predictions for variables using the test data
    bilstm_predictions = bilstm_model.predict(X_test)
    return bilstm_predictions, y_test


def build_cnn_bilstm(input_shape, output_dim):
    inputs = Input(shape=input_shape)

    # CNN - Extrage caracteristici
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)  # Transformă caracteristicile pentru LSTM

    # BiLSTM - Modelează dependențele temporale
    x = Reshape((1, x.shape[1]))(x)  # Reshape pentru LSTM
    x = Bidirectional(LSTM(64, return_sequences=False))(x)

    # Stratul final de predicție
    outputs = Dense(output_dim)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def predict_with_cnn_bilstm(target_cols):
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    X, y, scaler = preprocess_data(metars_df, target_cols)
    # split data into train and test information
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    input_shape = (X_train.shape[1], X_train.shape[2])  # (time, features)
    output_dim = len(target_cols)

    cnn_bilstm_model = build_cnn_bilstm(input_shape, output_dim)
    # Antrenare model
    cnn_bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    cnn_bilstm_predictions = cnn_bilstm_model.predict(X_test)
    return cnn_bilstm_predictions, y_test


# binary classification model for the presence of clouds
def build_cloud_presence_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# predict cloud nebulosity and altitude
def build_cloud_details_model(input_shape, output_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(output_dim)  # no activation function for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def predict_cloud_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(
        int)  # create a variable to indicate the presence of clouds
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    target_cloud_presence = ['cloud_presence']
    target_cloud_features = ['cloud_nebulosity', 'cloud_altitude']

    X, y, scaler = preprocess_data(metars_df, target_cloud_presence)
    # X = metars_df.drop(columns=[target_cloud_presence] + target_cloud_features)
    # y = metars_df[target_cloud_presence]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
    # y_train, y_test = y_train.astype('float32'), y_test.astype('float32')

    cloud_presence_model = build_cloud_presence_model((X_train.shape[1],))
    cloud_presence_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    cloud_presence_predictions = cloud_presence_model.predict(X)
    return cloud_presence_predictions, y_test


# cloud_presence_prediction, cloud_presence_test = predict_cloud_presence()
# print(f"Accuracy: {accuracy_score(cloud_presence_test, cloud_presence_prediction):.4f}")

# target_columns = ['wind_direction', 'wind_speed', 'predominant_horizontal_visibility', 'precipitation',
#                   'present_fog', 'cloud_nebulosity', 'cloud_altitude', 'air_temperature', 'dew_point', 'air_pressure']
# cnn_predictions, cnn_test = predict_with_cnn(target_columns)
target_columns = ['air_temperature', 'dew_point', 'air_pressure']
bilstm_predictions, bilstm_test = predict_with_bilstm(target_columns)  # 'air_temperature', 'dew_point', 'air_pressure'
# cnn_bilstm_predictions, cnn_bilstm_test = predict_with_cnn_bilstm(target_columns)


# for i, col in enumerate(target_columns):
#     print(f"\n1D-CNN Evaluation für {col}:")
#     print(f"MAE: {mean_absolute_error(cnn_test[:, i], cnn_predictions[:, i]):.4f}")
#     print(f"RMSE: {np.sqrt(mean_squared_error(cnn_test[:, i], cnn_predictions[:, i])):.4f}")
#     print(f"R² Score: {r2_score(cnn_test[:, i], cnn_predictions[:, i]):.4f}")

for i, col in enumerate(target_columns):
    print(f"\nBiLSTM Evaluation für {col}:")
    print(f"MAE: {mean_absolute_error(bilstm_test[:, i], bilstm_predictions[:, i]):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(bilstm_test[:, i], bilstm_predictions[:, i])):.4f}")
    print(f"R² Score: {r2_score(bilstm_test[:, i], bilstm_predictions[:, i]):.4f}")

# for i, col in enumerate(target_columns):
#     print(f"\nBiLSTM Evaluation für {col}:")
#     print(f"MAE: {mean_absolute_error(cnn_bilstm_test[:, i], cnn_bilstm_predictions[:, i]):.4f}")
#     print(f"RMSE: {np.sqrt(mean_squared_error(cnn_bilstm_test[:, i], cnn_bilstm_predictions[:, i])):.4f}")
#     print(f"R² Score: {r2_score(cnn_bilstm_test[:, i], cnn_bilstm_predictions[:, i]):.4f}")
