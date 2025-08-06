import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Energy (kWh)")
    plt.ylabel("Predicted Energy (kWh)")
    plt.title("Actual vs Predicted Energy Usage")
    plt.tight_layout()
    plt.show()

