import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from Data import csv_file_handler

metars_df = csv_file_handler.read_metar_df_from_csv_file()
metars_df['cloud_presence'] = (metars_df['cloud_nebulosity'] > 0).astype(
    int)  # create a variable to indicate the presence of clouds
metars_df.index = pd.to_datetime(metars_df['observation_time'], format="%Y-%m-%d %H:%M:%S")
X = metars_df[['cloud_presence']]
y = metars_df[['cloud_presence']]
# could predict cloud presence with an accuracy of 68% only using 'air_temperature', 'air_pressure', 'dew_point', 'precipitation'
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
y_pred = np.array(y_pred.astype(int)).flatten()
y_test = np.array(y_test).flatten()

train_result = pd.DataFrame(
    data={'Train Prediction': y_pred,
          'Actual Value': y_test})
with open('predictions/cloud_presence_prediction_xgb.txt', 'w') as f:
    f.write(train_result.to_string())
