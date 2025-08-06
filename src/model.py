import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("data/model_energy_data.csv")
X = df[["Num_Layers", "FLOPs_in_TFLOPs", "Training_Hours"]]
y = df["CO2_kg"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "co2_predictor.joblib")
print("✅ Model trained and saved as co2_predictor.joblib")