import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math

from Data import csv_file_handler

features = ['wind_dir_sin', 'wind_dir_cos', 'air_temperature', 'dew_point', 'air_pressure', 'hour', 'month',
            'precipitation', 'present_fog', 'cloud_nebulosity', 'cloud_altitude']
target = 'wind_speed'

df = csv_file_handler.read_metar_df_from_csv_file()
df['hour'] = pd.to_datetime(df['observation_time']).dt.hour
df['month'] = pd.to_datetime(df['observation_time']).dt.month
df['wind_dir_sin'] = np.sin(np.deg2rad(df['wind_direction']))
df['wind_dir_cos'] = np.cos(np.deg2rad(df['wind_direction']))

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i] < 0:
        y_pred[i] = 0
    else:
        frac, whole = math.modf(y_pred[i])
        if frac >= 0.5:
            y_pred[i] = int(whole + 1)
        else:
            y_pred[i] = int(whole)

print(f"R² Score: {r2_score(y_test, y_pred.astype(int)):.4f}")

# train_result = pd.DataFrame(
#     data={'Train Prediction': y_pred.astype(int),
#           'Actual Value': y_test})
# with open('predictions/wind_speed_xgb.txt', 'w') as f:
#     f.write(train_result.to_string())
