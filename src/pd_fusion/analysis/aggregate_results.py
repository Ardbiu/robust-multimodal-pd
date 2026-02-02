import pandas as pd
from pathlib import Path
import yaml
import argparse
from typing import List, Dict

def load_results(sweep_dir: Path) -> List[Dict]:
    results_list = []
    # Identify run directories
    # Structure: sweep_dir/model_seed/results.yaml
    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        res_file = run_dir / "results.yaml"
        agg_file = run_dir / "results_aggregated.yaml"
        if res_file.exists() or agg_file.exists():
            try:
                model_name = None
                seed = "unknown"

                prov_file = run_dir / "provenance.yaml"
                if prov_file.exists():
                    with open(prov_file, "r") as f:
                        prov = yaml.load(f, Loader=yaml.UnsafeLoader)
                    seed = prov.get("seed", seed)
                config_file = run_dir / "resolved_config.yaml"
                if config_file.exists():
                    with open(config_file, "r") as f:
                        conf = yaml.load(f, Loader=yaml.UnsafeLoader)
                    model_type = conf.get("model_type", None)
                    modality = conf.get("modality", None)
                    if model_type == "unimodal_gbdt" and modality:
                        model_name = f"unimodal_{modality}"
                    else:
                        model_name = model_type

                if model_name is None:
                    parts = run_dir.name.split("_s")
                    if len(parts) == 2:
                        model_name = parts[0]
                        seed = parts[1]
                    else:
                        model_name = run_dir.name

                if res_file.exists():
                    with open(res_file, "r") as f:
                        metrics = yaml.load(f, Loader=yaml.UnsafeLoader)
                    for scenario, values in metrics.items():
                        row = {
                            "Model": model_name,
                            "Seed": seed,
                            "Scenario": scenario,
                            "_from_cv": False,
                        }
                        row.update(values)
                        results_list.append(row)
                else:
                    with open(agg_file, "r") as f:
                        metrics = yaml.load(f, Loader=yaml.UnsafeLoader)
                    for scenario, values in metrics.items():
                        row = {
                            "Model": model_name,
                            "Seed": seed,
                            "Scenario": scenario,
                            "_from_cv": True,
                        }
                        for metric, stats in values.items():
                            row[f"{metric}_mean"] = stats.get("mean")
                            row[f"{metric}_std"] = stats.get("std")
                        results_list.append(row)
            except Exception as e:
                print(f"Error reading {run_dir}: {e}")
                
    return results_list

def main():
    parser = argparse.ArgumentParser(description="Aggregate sweep results")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Path to sweep directory (e.g. runs/sweep_TO_TIMESTAMP)")
    parser.add_argument("--output", type=str, default="summary.csv")
    args = parser.parse_args()
    
    sweep_path = Path(args.sweep_dir)
    print(f"Aggregating results from {sweep_path}")
    
    data = load_results(sweep_path)
    if not data:
        print("No results found.")
        return
        
    df = pd.DataFrame(data)
    
    # Save raw
    df.to_csv(Path(args.output), index=False)
    print(f"Saved raw results to {args.output}")
    
    if df["_from_cv"].any():
        summary_table = df.drop(columns=["_from_cv"])
        summary_csv = Path(args.output).with_name("summary_table.csv")
        summary_tex = Path(args.output).with_name("summary_table.tex")
        summary_table.to_csv(summary_csv, index=False)
        summary_table.to_latex(summary_tex, index=False, float_format="%.4f")
        print(f"Saved summary table to {summary_csv} and {summary_tex}")
    else:
        # Compute aggregates
        # Group by Model, Scenario -> Mean/Std of metrics
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        # Exclude Seed if numeric
        if "Seed" in numeric_cols: numeric_cols.remove("Seed")
        
        agg_df = df.groupby(["Model", "Scenario"])[numeric_cols].agg(["mean", "std"])
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        
        agg_file = Path(args.output).with_name("summary_aggregated.csv")
        agg_df.to_csv(agg_file)
        print(f"Saved aggregated results to {agg_file}")
        
        # Flat summary table for paper output
        summary_table = agg_df.reset_index()
        summary_csv = Path(args.output).with_name("summary_table.csv")
        summary_tex = Path(args.output).with_name("summary_table.tex")
        summary_table.to_csv(summary_csv, index=False)
        summary_table.to_latex(summary_tex, index=False, float_format="%.4f")
        print(f"Saved summary table to {summary_csv} and {summary_tex}")
    
    # Print minimal summary table (e.g. ROC-AUC mean for 'full_observation')
    print("\n--- Summary (Full Observation ROC-AUC) ---")
    try:
        subset = agg_df.xs("full_observation", level="Scenario")
        print(subset[["roc_auc_mean", "roc_auc_std"]].sort_values("roc_auc_mean", ascending=False))
    except Exception:
        print("Could not extract full_observation summary.")

if __name__ == "__main__":
    main()
