import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

MODEL_PATH = "model/anomaly_detector/iso_forest.pkl"

def generate_ref_data():
    np.random.seed(42)
    X = np.random.randn(200, 4)  # synthetic normal usage
    return X

def train_anomaly_detector():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    X = generate_ref_data()
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return clf

def flag_anomaly(input_features: list):
    clf = train_anomaly_detector()
    is_outlier = clf.predict([input_features])[0] == -1
    return is_outlier