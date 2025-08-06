import pandas as pd
from datetime import datetime, timedelta

def load_energy_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df: pd.DataFrame):
    # If 'timestamp' column doesn't exist, create synthetic timestamps
    if 'timestamp' not in df.columns:
        df['timestamp'] = [datetime.now() - timedelta(hours=i) for i in range(len(df))][::-1]
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df

