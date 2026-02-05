import argparse
import logging
from pathlib import Path
import pandas as pd
import yaml


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppmi_report")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)

        file_handler = logging.FileHandler(out_dir / "ppmi_eval_report.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPMI tabular report")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", required=True, help="Run directory with results_all.csv")
    args = parser.parse_args()

    _ = load_config(args.config)
    out_dir = Path(args.out_dir)
    logger = setup_logging(out_dir)

    results_path = out_dir / "results_all.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    df = pd.read_csv(results_path)
    if args.seed is not None:
        df = df[df["seed"] == args.seed]

    summary = df.groupby(["model", "ablation"]).agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary_path = out_dir / "summary_sweep_mean.csv"
    summary.to_csv(summary_path, index=False)

    ranking = summary.sort_values("roc_auc_mean", ascending=False)
    ranking_path = out_dir / "ranking_table.csv"
    ranking.to_csv(ranking_path, index=False)

    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved ranking to %s", ranking_path)


if __name__ == "__main__":
    main()
