import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import os

# Dummy Dataset with 4 Features: layers, hours, flops, complexity
X = np.random.rand(100, 4)  # 100 samples, 4 features each
y = np.random.rand(100) * 100  # Target: energy consumption (kWh)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Save the Model
os.makedirs('.', exist_ok=True)
joblib.dump(model, 'co2_predictor.joblib')
print("Model retrained and saved as 'co2_predictor.joblib' with 4 input features.")

