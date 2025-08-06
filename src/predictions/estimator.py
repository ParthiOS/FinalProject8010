import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "model/energy_predictor/rf_model.pkl"

# Sample training data (use real data if available)
def generate_training_data():
    np.random.seed(42)
    X = np.random.randint(1, 30, size=(100, 4))
    y = (X[:, 0] * X[:, 1] * X[:, 2] * 0.0005 + X[:, 3] * 0.1) + np.random.rand(100)
    return X, y

def train_energy_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    X, y = generate_training_data()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def predict_energy(model, input_dict):
    features = np.array([[input_dict["layers"],
                          input_dict["hours"],
                          input_dict["flops"],
                          input_dict["complexity"]]])
    return model.predict(features)[0]