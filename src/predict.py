import joblib
import numpy as np

def predict_energy(model_path, layers, hours, flops, complexity):
    model = joblib.load(model_path)
    input_features = np.array([[layers, hours, flops, complexity]])  # 4 Features
    prediction = model.predict(input_features)[0]
    return prediction

def flag_anomaly(anomaly_model_path, input_features):
    model = joblib.load(anomaly_model_path)
    input_features = np.array(input_features).reshape(1, -1)
    is_anomaly = model.predict(input_features)[0] == -1  # Isolation Forest returns -1 for anomaly
    return is_anomaly
