import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/logs/usage_log.csv", parse_dates=["timestamp"])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

plt.figure(figsize=(12, 6))
plt.plot(df.index, df.predicted_kwh, label="Predicted energy")
anoms = df[df.anomaly == True]
plt.scatter(anoms.index, anoms.predicted_kwh, color='red', label="Anomaly", zorder=5)
plt.xlabel("Time")
plt.ylabel("Predicted kWh")
plt.title("Energy Usage Over Time (with Anomalies)")
plt.legend()
plt.tight_layout()
plt.show()