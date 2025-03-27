import numpy as np
import pandas as pd
import csv_file_handler
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def preprocess_data(df, target_cols, sequence_length=10):
    df = df[target_cols]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length):
        X.append(df_scaled[i:i + sequence_length])
        y.append(df_scaled[i + sequence_length, [df.columns.get_loc(col) for col in target_cols]])

    return np.array(X), np.array(y), scaler


# binary classification model for the presence of fog
def build_fog_presence_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # sigmoid for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def predict_fog_presence():
    metars_df = csv_file_handler.read_metar_df_from_csv_file()
    metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")

    X, y, scaler = preprocess_data(metars_df, ['present_fog'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fog_presence_model = build_fog_presence_model((X_train.shape[1],))
    fog_presence_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    fog_presence_predictions = fog_presence_model.predict(X_test)
    return fog_presence_predictions, y_test


fog_presence_prediction, fog_presence_test = predict_fog_presence()
fog_presence_prediction_binary = (fog_presence_prediction > 0.5).astype(int)
fog_presence_test = fog_presence_test.astype(int)
print(f"Accuracy: {accuracy_score(fog_presence_test, fog_presence_prediction_binary):.4f}")

# train_result = pd.DataFrame(
#     data={'Train Prediction': fog_presence_prediction_binary.flatten(),
#           'Actual Value': fog_presence_test.flatten()})
# with open('predictions/fog_predictions.txt', 'w') as f:
#     f.write(train_result.to_string())
