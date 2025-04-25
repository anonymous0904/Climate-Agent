from flask import Flask, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

VAR_FILES = {
    "air_temperature": "air_temperature_predictions.csv",
    "air_pressure": "air_pressure_predictions.csv",
    "dew_point": "dew_point_predictions.csv",
    "cloud_altitude": "cloud_altitude_predictions.csv",
    "cloud_nebulosity": "cloud_nebulosity_predictions.csv",
    "present_fog": "fog_predictions.csv",
    "precipitation": "precipitation_predictions.csv",
    "predominant_horizontal_visibility": "visibility_predictions.csv",
    "wind_direction": "wind_direction_prediction.csv",
    "wind_speed": "wind_speed_predictions.csv"
}


@app.route('/api/<variable>')
def get_data(variable):
    if variable not in VAR_FILES:
        return jsonify({"error": "Invalid variable"}), 404
    df = pd.read_csv(f'Data/predictions/{VAR_FILES[variable]}')
    df['Time'] = pd.to_datetime(df['Time'])
    # df['Train Prediction'] = pd.to_numeric(df['Train Prediction'], errors='coerce')
    # df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df = df[['Time', 'Actual Value', 'Train Prediction']]
    return df.to_dict(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
