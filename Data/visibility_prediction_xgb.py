import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Data import csv_file_handler

metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")

X = metars_df[['predominant_horizontal_visibility', 'air_temperature', 'present_fog', 'air_pressure', 'dew_point',
               'precipitation']]
y = metars_df[['predominant_horizontal_visibility']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i] < 800:
        vis = y_pred[i]
        y_pred[i] = (vis // 50) * 50
    elif 800 < y_pred[i] < 5000:
        vis = y_pred[i]
        y_pred[i] = (vis // 100) * 100
    elif 5000 < y_pred[i] < 9950:
        vis = y_pred[i]
        y_pred[i] = (vis // 1000) * 1000
    elif 9950 <= y_pred[i]:
        y_pred[i] = 9999

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

y_pred = np.array(y_pred.astype(int)).flatten()
y_test = np.array(y_test).flatten()

# train_result = pd.DataFrame(
#     data={'Train Prediction': y_pred,
#           'Actual Value': y_test})
# with open('predictions/visibility_predictions_xgb.txt', 'w') as f:
#     f.write(train_result.to_string())
