import os
import csv
from datetime import datetime

LOGFILE = "data/logs/usage_log.csv"

def log_usage(record: dict):
    header = ["timestamp", "prompt", "layers", "hours", "flops", "complexity", "predicted_kwh", "anomaly"]
    os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)
    write_header = not os.path.exists(LOGFILE)
    with open(LOGFILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        row = {"timestamp": datetime.utcnow().isoformat(), **record}
        writer.writerow(row)