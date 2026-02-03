import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from pd_fusion.utils.metrics import compute_metrics
import yaml

def _get_model_name(run_dir: Path) -> str:
    prov = run_dir / "resolved_config.yaml"
    if prov.exists():
        conf = yaml.safe_load(open(prov))
        model_type = conf.get("model_type", None)
        modality = conf.get("modality", None)
        if model_type == "unimodal_gbdt" and modality:
            return f"unimodal_{modality}"
        return model_type or run_dir.name
    parts = run_dir.name.split("_s")
    return parts[0] if len(parts) == 2 else run_dir.name

def _bootstrap_metrics(y_true, y_prob, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    metrics = []
    for _ in range(n):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        m = compute_metrics(y_true[sample_idx], y_prob[sample_idx])
        metrics.append(m)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs from per-fold predictions")
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--group-col", type=str, default="")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    model_preds = {}

    for run_dir in sweep_dir.iterdir():
        if not run_dir.is_dir():
            continue
        pred_files = list(run_dir.glob("preds_fold_*_full_observation.csv"))
        if not pred_files:
            continue
        df_preds = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
        model_name = _get_model_name(run_dir)
        model_preds.setdefault(model_name, []).append(df_preds)

    rows = []
    for model, dfs in model_preds.items():
        df = pd.concat(dfs, ignore_index=True)
        y_true = df["y_true"].values
        y_prob = df["y_prob"].values

        if args.group_col and args.group_col in df.columns:
            g = df.groupby(args.group_col).agg({"y_true": "first", "y_prob": "mean"}).reset_index()
            y_true = g["y_true"].values
            y_prob = g["y_prob"].values

        boot = _bootstrap_metrics(y_true, y_prob, n=args.n)
        for metric in boot[0].keys():
            vals = [b[metric] for b in boot]
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append({
                "Model": model,
                "Metric": metric,
                "CI_low": float(lo),
                "CI_high": float(hi),
            })

    out_path = sweep_dir / "summary_bootstrap_ci.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved bootstrap CIs to {out_path}")

if __name__ == "__main__":
    main()
