import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
import os

# Generate Dummy Data for Anomaly Detection
# Features: layers, hours, flops, complexity
X = np.random.rand(200, 4) * [48, 48, 1000, 100]  # Scale realistic ranges

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X)

# Save the Anomaly Detection Model
joblib.dump(iso_forest, 'iso_forest.pkl')
print("Anomaly Detection Model saved as 'iso_forest.pkl'.")
