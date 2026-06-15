import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- Paths ---
DATA_PATH = "data_curated/panel_for_model.csv"  # your dataset
SCALER_PATH = "models/markov_scaler.pkl"       # path to save scaler

# --- Load data ---
data = pd.read_csv(DATA_PATH)

# --- Select only numeric columns for scaling ---
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
X_train = data[numeric_cols].values

# --- Fit scaler ---
scaler = StandardScaler()
scaler.fit(X_train)

# --- Save scaler ---
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_PATH)

print(f"Scaler rebuilt and saved at {SCALER_PATH}")
