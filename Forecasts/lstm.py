from Data import DatabaseHandler
import keras.src.models
from keras.src.layers import *
from keras.src.callbacks import ModelCheckpoint
from keras.src.losses import MeanSquaredError
from keras.src.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def plot_temperature_matplotlib():
    metars = DatabaseHandler.DatabaseHandler().get_metars_df()
    metars.index = pd.to_datetime(metars['observation_time'], format="%Y-%m-%d %H:%M:%S")
    air_temperature = metars['air_temperature']
    plt.figure(figsize=(10, 5))
    plt.plot(air_temperature)
    plt.title('Air temperature over time')
    plt.xlabel('Time')
    plt.ylabel('Air temperature')
    plt.show()


def plot_temperature_plotly():
    metars = DatabaseHandler.DatabaseHandler().get_metars_df()
    metars.index = pd.to_datetime(metars['observation_time'], format="%Y-%m-%d %H:%M:%S")
    fig = px.line(metars, x=metars.index, y='air_temperature', title='Air Temperature Over Time')
    fig.show()


# # [[ [1], [1.5], [2], [2.5], [3] ]]    [3.5]
# # [[ [1.5], [2], [2.5], [3], [3.5] ]]  [4]
# # [[ [2], [2.5], [3], [3.5], [4] ]]    [4.5]
# def df_to_X_y(df, window_size=5):
#     df_as_np = df.to_numpy()
#     X = []  # matrix of input data, the values of the air temperature every half an hour
#     y = []  # output data, the actual temperature values that we compare with the forecasted ones
#     for i in range(len(df_as_np) - window_size):
#         row = [[t] for t in df_as_np[i:i + 5]]
#         X.append(row)
#         label = df_as_np[i + 5]  # the actual value of the temperature that we predict using the last 5 values
#         y.append(label)
#
#     return np.array(X), np.array(y)


# metars_df = DatabaseHandler.DatabaseHandler().get_metars_df()
# metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
# air_temperature = metars_df['air_temperature']
# WINDOW_SIZE = 5
# X, y = df_to_X_y(air_temperature, WINDOW_SIZE)
#
# X_train, y_train = X[:40000], y[:40000]
# X_val, y_val = X[40000:45000], y[40000:45000]
# X_test, y_test = X[45000:], y[45000:]
#
# model1 = keras.models.Sequential()
# model1.add(InputLayer((5, 1)))
# model1.add(LSTM(64))
# model1.add(Dense(8, 'relu'))
# model1.add(Dense(1, 'linear'))
#
# # model1.summary()
#
# # cp = ModelCheckpoint('model1/model.h5', save_best_only=True)
# # model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
# # model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])
#
#
# model1 = keras.models.load_model('model1/model.h5')
# train_predictions = model1.predict(X_train).flatten()
# train_result = pd.DataFrame(data={'Train Prediction': train_predictions, 'Actual Value': y_train})
# with open('model1/train_result.txt', 'w') as f:
#     f.write(train_result.to_string())


# initially, this class predicts air_temperature only using metar data
class LSTM:
    def __init__(self):
        self.metars_df = DatabaseHandler.DatabaseHandler().get_metars_df()
        self.tafs_df = DatabaseHandler.DatabaseHandler().get_tafs_df()

    # [[ [1], [1.5], [2], [2.5], [3] ]]    [3.5]
    # [[ [1.5], [2], [2.5], [3], [3.5] ]]  [4]
    # [[ [2], [2.5], [3], [3.5], [4] ]]    [4.5]
    # transform data frame into input and output data format used to make predictions
    def df_to_X_y(self, variables, window_size=5):
        df_as_np = variables.to_numpy()
        X = []  # matrix of input data, the values of the air temperature every half an hour
        y = []  # output data, the actual temperature values that we compare with the forecasted ones
        for i in range(len(df_as_np) - window_size):
            row = [[t] for t in df_as_np[i:i + 5]]
            X.append(row)
            label = df_as_np[i + 5]  # the actual value of the temperature that we predict using the last 5 values
            y.append(label)

        return np.array(X), np.array(y)

    # returns data about air temperature from the data frame
    def filter_air_temperature(self):
        self.metars_df.index = pd.to_datetime(self.metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
        air_temperature = self.metars_df['air_temperature']
        return air_temperature

    def predict_model(self):
        air_temperature = self.filter_air_temperature()
        X, y = self.df_to_X_y(air_temperature, 5)

        X_train, y_train = X[:40000], y[:40000]
        X_val, y_val = X[40000:45000], y[40000:45000]
        X_test, y_test = X[45000:], y[45000:]

        model = keras.models.Sequential()
        model.add(InputLayer((5, 1)))
        model.add(LSTM(64))
        model.add(Dense(8, 'relu'))
        model.add(Dense(1, 'linear'))

        cp = ModelCheckpoint('model1/model.h5', save_best_only=True)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

        return model

    def save_model_to_file(self):
        pass
