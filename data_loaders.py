import pandas as pd


def load_air_pressure_predictions():
    df = pd.read_csv('Data/predictions/air_pressure_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_air_temperature_predictions():
    df = pd.read_csv('Data/predictions/air_temperature_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_dew_point_predictions():
    df = pd.read_csv('Data/predictions/dew_point_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_cloud_altitude_predictions():
    df = pd.read_csv('Data/predictions/cloud_altitude_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_cloud_nebulosity_predictions():
    df = pd.read_csv('Data/predictions/cloud_nebulosity_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_fog_predictions():
    df = pd.read_csv('Data/predictions/fog_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_precipitation_predictions():
    df = pd.read_csv('Data/predictions/precipitation_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_visibility_predictions():
    df = pd.read_csv('Data/predictions/visibility_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_wind_direction_predictions():
    df = pd.read_csv('Data/predictions/wind_direction_prediction.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df


def load_wind_speed_predictions():
    df = pd.read_csv('Data/predictions/wind_speed_predictions.csv')
    df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Time'] = pd.to_datetime(df['Time'])
    return df
