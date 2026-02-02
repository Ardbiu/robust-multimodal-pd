import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

def save_plot_data(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)

def plot_degradation_curve(results: dict, output_path: Path):
    """
    Plots performance (ROC-AUC) vs scenario.
    """
    data = []
    for name, metrics in results.items():
        data.append({"Scenario": name, "ROC-AUC": metrics["roc_auc"], "PR-AUC": metrics["pr_auc"]})
    
    df = pd.DataFrame(data)
    save_plot_data(df, output_path.with_suffix(".csv"))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Scenario", y="ROC-AUC", hue="Scenario")
    plt.title("Model Robustness: ROC-AUC vs Missingness Scenario")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_calibration_curve_func(y_true, y_prob, output_path: Path, model_name="Model"):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    # Save data
    df = pd.DataFrame({"Mean_Predicted_Probability": prob_pred, "Fraction_of_Positives": prob_true})
    save_plot_data(df, output_path.with_suffix(".csv"))
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.title(f"Reliability Diagram ({model_name})")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, output_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    save_plot_data(df, output_path.with_suffix(".csv"))
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_pr_curve(y_true, y_prob, output_path: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    df = pd.DataFrame({"Recall": recall, "Precision": precision})
    save_plot_data(df, output_path.with_suffix(".csv"))
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="PR Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_risk_coverage(data: dict, output_path: Path):
    """
    Plots Risk vs Coverage curve.
    data: {"coverage": np.array, "risk": np.array}
    """
    coverage = data["coverage"]
    risk = data["risk"]
    
    # Save data
    df = pd.DataFrame({"Coverage": coverage, "Risk": risk})
    save_plot_data(df, output_path.with_suffix(".csv"))

    plt.figure(figsize=(6, 6))
    plt.plot(coverage, risk, label="Risk-Coverage")
    plt.xlabel("Coverage (Fraction of samples predicted)")
    plt.ylabel("Risk (Error Rate)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    # plt.ylim(0, max(risk)*1.1 if len(risk)>0 else 1)
    plt.legend()
    plt.savefig(output_path)
    plt.close()
