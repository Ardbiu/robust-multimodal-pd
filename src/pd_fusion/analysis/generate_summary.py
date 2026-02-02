import argparse
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_results(run_dir):
    path = Path(run_dir) / "results_aggregated.yaml"
    if not path.exists():
        logging.warning(f"No results found in {run_dir}")
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_summary(run_dirs, output_dir, metric="roc_auc", scenario="random_1_drop"):
    records = []
    
    for rd in run_dirs:
        data = load_results(rd)
        if not data: continue
        
        # Infer model name from directory name or content
        # Run dir expected format: runs/cv_modelname or similar
        model_name = Path(rd).name.replace("cv_", "").replace("run_", "")
        
        # Flatten metrics
        # Structure: {scenario: {metric: {mean: x, std: y}}}
        for scen, metrics in data.items():
            for met, stats in metrics.items():
                records.append({
                    "Model": model_name,
                    "Scenario": scen,
                    "Metric": met,
                    "Mean": stats["mean"],
                    "Std": stats["std"]
                })

    df = pd.DataFrame(records)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save Full CSV
    df.to_csv(out_path / "final_benchmark_summary.csv", index=False)
    
    # Pivot for Table (Focus: ROC-AUC, Balanced Acc)
    # Rows: Model, Cols: Scenario (Full, Random Drop)
    # Value: "Mean ± Std"
    
    def format_val(row):
        return f"{row['Mean']:.3f} ± {row['Std']:.3f}"
    
    df["Formatted"] = df.apply(format_val, axis=1)
    
    pivot_df = df.pivot(index="Model", columns=["Metric", "Scenario"], values="Formatted")
    
    # Filter columns if possible to key ones
    # E.g. (roc_auc, full_observation), (roc_auc, random_1_drop)
    cols_to_keep = []
    for m in ["roc_auc", "balanced_accuracy"]:
        for s in ["full_observation", "random_1_drop", "clinical_only"]:
            if (m, s) in pivot_df.columns:
                cols_to_keep.append((m, s))
                
    if cols_to_keep:
        pivot_df = pivot_df[cols_to_keep]
        
    pivot_df.to_latex(out_path / "summary_table.tex")
    logging.info(f"Saved summary table to {out_path / 'summary_table.tex'}")

    # Plot Comparison (Bar Chart for Degradation)
    # Filter for specific metric
    subset = df[(df["Metric"] == metric) & (df["Scenario"].isin(["full_observation", scenario]))]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x="Model", y="Mean", hue="Scenario", capsize=0.1)
    plt.title(f"Model Robustness: {metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path / "robustness_comparison.png")
    logging.info(f"Saved plot to {out_path / 'robustness_comparison.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="List of run directories")
    parser.add_argument("--output", default="final_results", help="Output directory")
    args = parser.parse_args()
    
    setup_logging()
    generate_summary(args.runs, args.output)
