import DatabaseHandler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


# # Set the Matplotlib backend to TkAgg
# plt.switch_backend('TkAgg')


# currently not working
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


# [[ [1], [1.5], [2], [2.5], [3] ]]    [3.5]
# [[ [1.5], [2], [2.5], [3], [3.5] ]]  [4]
# [[ [2], [2.5], [3], [3.5], [4] ]]    [4.5]
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []  # matrix of input data, the values of the air temperature every half an hour
    y = []  # output data, the actual temperature values that we compare with the forecasted ones
    for i in range(len(df_as_np) - window_size):
        row = [[t] for t in df_as_np[i:i + 5]]
        X.append(row)
        label = df_as_np[i + 5]  # the actual value of the temperature that we predict using the last 5 values
        y.append(label)

    return np.array(X), np.array(y)


metars_df = DatabaseHandler.DatabaseHandler().get_metars_df()
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
air_temperature = metars_df['air_temperature']
WINDOW_SIZE = 5
X, y = df_to_X_y(air_temperature, WINDOW_SIZE)

X_train, y_train = X[:40000], y[:40000]
X_val, y_val = X[40000:45000], y[40000:45000]
X_test, y_test = X[45000:], y[45000:]

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

# model1.summary()

cp = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])


class LSTM:
    def __init__(self):
        self.metars_df = DatabaseHandler.DatabaseHandler().get_metars_df()
        self.tafs_df = DatabaseHandler.DatabaseHandler().get_tafs_df()
