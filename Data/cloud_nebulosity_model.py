import numpy as np
import pandas as pd
from keras import Sequential, Input
from keras.src.layers import LSTM, Bidirectional, Dropout, Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from Data import csv_file_handler


def preprocess_data(df, target_cols, sequence_length=10):
    # df = df.drop(columns=['observation_time'])
    df = df[target_cols]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length, [df.columns.get_loc(col) for col in target_cols]])

    return np.array(X), np.array(y), scaler


def build_cloud_nebulosity_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input shape will be (sequence_length, num_features)
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # 5 classes (0, 1, 2, 3, 4)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(
    int)  # create a variable to indicate the presence of clouds
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
X, y, scaler = preprocess_data(metars_df, ['cloud_nebulosity'])
y = y[:, 0].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cloud_nebulosity_model = build_cloud_nebulosity_model((X_train.shape[1], X_train.shape[2]))
cloud_nebulosity_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
cloud_nebulosity_predictions = cloud_nebulosity_model.predict(X_test)
cloud_nebulosity_predictions = np.argmax(cloud_nebulosity_predictions, axis=1)
print(f"Accuracy: {accuracy_score(cloud_nebulosity_predictions, y_test):.4f}")
