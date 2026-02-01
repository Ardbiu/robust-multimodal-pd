import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_degradation_curve(results, output_path: Path):
    """
    Plots performance vs scenario.
    """
    df = pd.DataFrame(results).T
    plt.figure(figsize=(10, 6))
    df["roc_auc"].plot(kind="bar")
    plt.title("Model Robustness under Missingness Scenarios")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_reliability_diagram(y_true, y_prob, output_path: Path):
    # Skeleton
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("Reliability Diagram")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.savefig(output_path)
    plt.close()
活跃的
