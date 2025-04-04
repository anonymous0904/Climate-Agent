import random

import numpy as np
import pandas as pd
from keras import Input
from keras.src.callbacks import EarlyStopping
import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, Bidirectional, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def preprocess_data(df, target_cols, sequence_length=10):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length, [df.columns.get_loc(col) for col in target_cols]])

    return np.array(X), np.array(y), scaler


def build_bilstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def predict_visibility_with_bilstm():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    metars_df = metars_df[['predominant_horizontal_visibility']]
    X, y, scaler = preprocess_data(metars_df, ['predominant_horizontal_visibility'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')
    bilstm_model = build_bilstm_model(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16,
                     callbacks=[early_stopping])
    bilstm_predictions = bilstm_model.predict(X_test)
    bilstm_predictions = bilstm_predictions.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return scaler.inverse_transform(bilstm_predictions).astype(int), scaler.inverse_transform(y_test).astype(int)


visibility_predictions, visibility_test = predict_visibility_with_bilstm()
for i in range(len(visibility_predictions)):
    if visibility_predictions[i] < 800:
        vis = visibility_predictions[i]
        visibility_predictions[i] = (vis // 50) * 50
    elif 800 < visibility_predictions[i] < 5000:
        vis = visibility_predictions[i]
        visibility_predictions[i] = (vis // 100) * 100
    elif 5000 < visibility_predictions[i] < 9999:
        vis = visibility_predictions[i]
        visibility_predictions[i] = (vis // 1000) * 1000
    elif 9999 < visibility_predictions[i]:
        visibility_predictions[i] = 9999

print(f"R² Score: {r2_score(visibility_test, visibility_predictions):.4f}")

# train_result = pd.DataFrame(
#     data={'Train Prediction': visibility_predictions.flatten(),
#           'Actual Value': visibility_test.flatten()})
# with open('predictions/visibility_predictons.txt', 'w') as f:
#     f.write(train_result.to_string())
