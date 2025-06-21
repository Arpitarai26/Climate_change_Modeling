# === 1. IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# === 2. LOAD DATASET ===
df = pd.read_csv("data/climate_data.csv")  # Adjust path if needed
print("Dataset loaded. Shape:", df.shape)

# === 3. CHECK FOR NULLS & CLEAN ===
print("Missing values:\n", df.isnull().sum())
df.dropna(inplace=True)

# === 4. DEFINE FEATURES AND TARGET ===
X = df[['CO2', 'sea_level', 'solar_radiation']]
y = df['temperature_anomaly']

# === 5. SCALE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 6. SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 7. TRAIN MODEL ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 8. EVALUATE MODEL ===
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# === 9. SAVE MODEL AND SCALER ===
os.makedirs("app/model", exist_ok=True)
joblib.dump(model, "app/model/model.pkl")
joblib.dump(scaler, "app/model/scaler.pkl")

print("Model and scaler saved successfully.")
