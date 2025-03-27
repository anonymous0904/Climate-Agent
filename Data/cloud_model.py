import numpy as np
import pandas as pd
import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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


def preprocess_cloud_details_data():
    pass


# binary classification model for the presence of clouds
def build_cloud_presence_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cloud_altitude_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)  # regression model
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_cloud_nebulosity_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')  # 5 classes (0, 1, 2, 3, 4)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_cloud_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(
        int)  # create a variable to indicate the presence of clouds
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    target_cloud_presence = ['cloud_presence']

    X, y, scaler = preprocess_data(metars_df, target_cloud_presence)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cloud_presence_model = build_cloud_presence_model((X_train.shape[1],))
    cloud_presence_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    cloud_presence_predictions = cloud_presence_model.predict(X_test)
    return cloud_presence_predictions, y_test


# predict the cloud cover and altitude when there are present clouds predicted
# use the output from the cloud presence prediction model
def predict_cloud_details(X_test, cloud_presence_predictions):
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(
        int)  # create a variable to indicate the presence of clouds
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
    cloud_presence_binary = (cloud_presence_predictions > 0.5).astype(int)


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


cloud_presence_prediction, cloud_presence_test = predict_cloud_presence()
cloud_presence_prediction_binary = (cloud_presence_prediction > 0.5).astype(int)
cloud_presence_test = cloud_presence_test.astype(int)
print(f"Accuracy: {accuracy_score(cloud_presence_test, cloud_presence_prediction_binary):.4f}")
# Accuracy: 0.8972

# train_result = pd.DataFrame(
#     data={'Train Prediction': cloud_presence_prediction_binary.flatten(),
#           'Actual Value': cloud_presence_test.flatten()})
# with open('predictions/cloud_presence_prediction.txt', 'w') as f:
#     f.write(train_result.to_string())
