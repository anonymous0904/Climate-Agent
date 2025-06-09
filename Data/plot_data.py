import matplotlib.pyplot as plt
import DatabaseHandler
import csv_file_handler
import pandas as pd
import seaborn as sns


def plot_data_availability():
    df = DatabaseHandler.DatabaseHandler().get_metars_df()[5290::2].sort_values('id')
    df = df[[
        'observation_time', 'wind_direction', 'wind_speed', 'predominant_horizontal_visibility', 'present_phenomena_1',
        'cloud_nebulosity_1', 'cloud_altitude_1', 'air_temperature', 'dew_point', 'air_pressure']]
    plt.figure(figsize=(10, 3))
    sns.heatmap(df.notnull(), cmap='Greens', cbar=False, yticklabels=False)
    plt.title("Fehlende Daten")
    plt.show()


def plot_missing_wind_direction():
    df = DatabaseHandler.DatabaseHandler().get_metars_df()[5290::2].sort_values('id')
    df = df[['wind_direction']]
    plt.figure(figsize=(3, 5))
    sns.heatmap(df.notnull(), cmap='Greens', cbar=False, yticklabels=False)
    plt.title("Fehlende Daten für Windrichtung")
    plt.show()


def plot_temperature():
    metars = csv_file_handler.read_metar_df_from_csv_file()
    metars.index = pd.to_datetime(metars['observation_time'], format="%Y-%m-%d %H:%M:%S")
    air_temperature = metars['air_temperature']
    plt.figure(figsize=(10, 5))
    plt.plot(air_temperature)
    plt.title('Lufttemperatur im Laufe der Zeit')
    plt.xlabel('Zeit')
    plt.ylabel('Lufttemperatur (°C)')
    plt.show()


def plot_wind_direction():
    metars = csv_file_handler.read_metar_df_from_csv_file()
    metars.index = pd.to_datetime(metars['observation_time'], format="%Y-%m-%d %H:%M:%S")
    air_temperature = metars['wind_direction']
    plt.figure(figsize=(15, 5))
    plt.plot(air_temperature)
    plt.title('Windrichtung im Laufe der Zeit')
    plt.xlabel('Zeit')
    plt.ylabel('Windrichtung')
    plt.show()


def plot_wind_speed():
    metars = csv_file_handler.read_metar_df_from_csv_file()
    metars.index = pd.to_datetime(metars['observation_time'], format="%Y-%m-%d %H:%M:%S")
    air_temperature = metars['wind_speed']
    plt.figure(figsize=(15, 5))
    plt.plot(air_temperature)
    plt.title('Windgeschwindigkeit im Laufe der Zeit')
    plt.xlabel('Zeit')
    plt.ylabel('Windgeschwindigkeit (kt)')
    plt.show()


def plot_visibility():
    metars = csv_file_handler.read_metar_df_from_csv_file()
    metars.index = pd.to_datetime(metars['observation_time'], format="%Y-%m-%d %H:%M:%S")
    air_temperature = metars['predominant_horizontal_visibility']
    plt.figure(figsize=(10, 5))
    plt.plot(air_temperature)
    plt.title('Sichtweite im Laufe der Zeit')
    plt.xlabel('Zeit')
    plt.ylabel('Sichtweite')
    plt.show()


def plot_air_pressure():
    metars = csv_file_handler.read_metar_df_from_csv_file()
    metars.index = pd.to_datetime(metars['observation_time'], format="%Y-%m-%d %H:%M:%S")
    air_temperature = metars['air_pressure']
    plt.figure(figsize=(10, 5))
    plt.plot(air_temperature)
    plt.title('Luftdruck im Laufe der Zeit')
    plt.xlabel('Zeit')
    plt.ylabel('Luftdruck (hPa)')
    plt.show()
