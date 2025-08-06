
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# Load usage log CSV
# ---------------------------
df = pd.read_csv("data/logs/usage_log.csv", parse_dates=["timestamp"])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# ---------------------------
# Create Plot
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df.predicted_kwh, label="Predicted energy")
anoms = df[df.anomaly == True]
plt.scatter(anoms.index, anoms.predicted_kwh, color='red', label="Anomaly", zorder=5)

# ---------------------------
# Customize Plot
# ---------------------------
plt.xlabel("Time")
plt.ylabel("Predicted kWh")
plt.title("Energy Usage Over Time (with Anomalies)")
plt.legend()
plt.tight_layout()

# ---------------------------
# Save to folder
# ---------------------------
output_path = "reports/visualizations/graphs"
os.makedirs(output_path, exist_ok=True)
plot_file = os.path.join(output_path, "energy_usage_plot.png")
plt.savefig(plot_file)

print(f"Plot saved to {plot_file}")