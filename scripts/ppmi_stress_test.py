import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pd_fusion.utils.metrics import compute_metrics


ID_COLS = {"subject_id", "visit_id", "visit_month", "date"}
GLOBAL_EXCLUDE = [
    r"^.*date.*$",
    r"^.*time.*$",
    r"^.*event.*$",
    r"^.*protocol.*$",
    r"^.*dose.*$",
    r"^.*site.*$",
    r"^.*center.*$",
    r"^.*scanner.*$",
    r"^.*acq.*$",
    r"^.*acquisition.*$",
    r"^.*series.*$",
    r"^.*version.*$",
    r"^.*reason.*$",
    r"^.*not_analyzed.*$",
    r"^.*notanalyzed.*$",
]

NONMOTOR_PATTERNS = [
    r"moca",
    r"cognition",
    r"sleep",
    r"epworth",
    r"rbd",
    r"rem",
    r"depress",
    r"gds",
    r"bdi",
    r"anxiety",
    r"stai",
    r"mood",
    r"upsit",
    r"smell",
    r"autonomic",
]

DATSBR_PATTERNS = [
    r"datscan",
    r"sbr",
    r"putamen",
    r"caudate",
    r"striat",
    r"asym",
]

MRI_PATTERNS = [
    r"mri_derived__",
    r"thickness",
    r"cortical",
    r"volume",
    r"area",
    r"aseg",
    r"hippo",
    r"entorhinal",
    r"amygdala",
    r"caudate",
    r"putamen",
    r"pallid",
    r"thalam",
    r"accumbens",
]


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppmi_stress")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(out_dir / "ppmi_stress_test.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def select_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce")


def filter_cols(cols: List[str], patterns: List[str]) -> List[str]:
    return [c for c in cols if any(re.search(p, c, re.IGNORECASE) for p in patterns)]


def exclude_cols(cols: List[str], patterns: List[str]) -> List[str]:
    return [c for c in cols if not any(re.search(p, c, re.IGNORECASE) for p in patterns)]


def get_all_numeric(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in ID_COLS and c != "label"]
    cols = exclude_cols(cols, GLOBAL_EXCLUDE)
    num_df = select_numeric(df, cols)
    return [c for c in num_df.columns if num_df[c].notna().any()]


def build_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_cols = get_all_numeric(df)
    nonmotor = filter_cols(all_cols, NONMOTOR_PATTERNS)
    datsbr = filter_cols(all_cols, DATSBR_PATTERNS)
    mri = filter_cols(all_cols, MRI_PATTERNS)
    imaging = sorted(set(datsbr + mri))
    return {
        "clinical": nonmotor,
        "imaging": imaging,
        "full": sorted(set(nonmotor + imaging)),
        "datsbr": datsbr,
        "mri": mri,
    }


def mask_features(X: np.ndarray, groups: Dict[str, List[int]], drop: Dict[str, bool]) -> np.ndarray:
    X_masked = X.copy()
    for name, idxs in groups.items():
        if drop.get(name, False) and idxs:
            X_masked[:, idxs] = 0.0
    return X_masked


def train_moddrop_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: Dict[str, List[int]],
    moddrop_prob: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise RuntimeError("PyTorch required for moddrop MLP") from exc

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    n_modalities = len([k for k in groups.keys() if k in ["clinical", "imaging"]])

    class MLP(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = MLP(n_features + n_modalities)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            # sample modality dropout
            keep_clin = (torch.rand(len(xb)) > moddrop_prob).float().unsqueeze(1)
            keep_img = (torch.rand(len(xb)) > moddrop_prob).float().unsqueeze(1)
            x_mod = xb.clone()
            if groups["clinical"]:
                x_mod[:, groups["clinical"]] *= keep_clin
            if groups["imaging"]:
                x_mod[:, groups["imaging"]] *= keep_img
            mask_vec = torch.cat([keep_clin, keep_img], dim=1)
            x_in = torch.cat([x_mod, mask_vec], dim=1)

            opt.zero_grad()
            logits = model(x_in)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    return model


def predict_moddrop(model, X: np.ndarray, groups: Dict[str, List[int]], drop: Dict[str, bool]):
    import torch
    X_masked = mask_features(X, groups, drop)
    keep_clin = 0.0 if drop.get("clinical", False) else 1.0
    keep_img = 0.0 if drop.get("imaging", False) else 1.0
    mask_vec = np.tile(np.array([keep_clin, keep_img], dtype=np.float32), (len(X_masked), 1))
    X_in = np.concatenate([X_masked, mask_vec], axis=1)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_in).float())
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def get_lgbm(num_threads: int, seed: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        num_threads=num_threads,
        force_col_wise=True,
        random_state=seed,
        class_weight="balanced",
    )


def main():
    parser = argparse.ArgumentParser(description="PPMI stress test for missing clinical data")
    parser.add_argument(
        "--input-csv",
        default="/home/adixit1/IEEE-spid/data/processed/ppmi/ppmi_subject_baseline.csv",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--moddrop-prob", type=float, default=0.3)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"runs/ppmi_stress_test_{timestamp}")
    logger = setup_logging(out_dir)

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_threads)

    df = pd.read_csv(args.input_csv, low_memory=False)
    df = df.dropna(subset=["label"]).copy()

    groups = build_groups(df)
    if not groups["clinical"] or not groups["imaging"]:
        raise ValueError("Need both clinical (non-motor) and imaging features for stress test")

    feature_cols = groups["full"]
    X = select_numeric(df, feature_cols)
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # map feature indices per group
    col_index = {c: i for i, c in enumerate(feature_cols)}
    group_idx = {
        "clinical": [col_index[c] for c in groups["clinical"] if c in col_index],
        "imaging": [col_index[c] for c in groups["imaging"] if c in col_index],
    }

    y = df["label"].values.astype(int)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    scenarios = {
        "full": {"clinical": False, "imaging": False},
        "missing_clinical": {"clinical": True, "imaging": False},
        "missing_imaging": {"clinical": False, "imaging": True},
    }

    rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), start=1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Baseline LGBM
        lgbm = get_lgbm(args.num_threads, args.seed + fold)
        lgbm.fit(X_train, y_train)

        # ModDrop MLP
        model = train_moddrop_mlp(
            X_train, y_train, group_idx, args.moddrop_prob, args.epochs, args.batch_size, 1e-3, args.seed + fold
        )

        for scen_name, drop in scenarios.items():
            # baseline
            X_test_masked = mask_features(X_test, group_idx, drop)
            p_lgbm = lgbm.predict_proba(X_test_masked)[:, 1]
            metrics_lgbm = compute_metrics(y_test, p_lgbm)
            rows.append({
                "model": "lgbm",
                "scenario": scen_name,
                "fold": fold,
                **metrics_lgbm,
            })

            # moddrop
            p_mod = predict_moddrop(model, X_test, group_idx, drop)
            metrics_mod = compute_metrics(y_test, p_mod)
            rows.append({
                "model": "moddrop_mlp",
                "scenario": scen_name,
                "fold": fold,
                **metrics_mod,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "stress_test_per_fold.csv", index=False)

    summary = out_df.groupby(["model", "scenario"]).agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out_dir / "stress_test_summary.csv", index=False)

    # Plot (ROC-AUC bar)
    try:
        import matplotlib.pyplot as plt
        plot_df = summary.copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, model in enumerate(plot_df["model"].unique()):
            subset = plot_df[plot_df["model"] == model]
            ax.bar(
                np.arange(len(subset)) + i * 0.35,
                subset["roc_auc_mean"],
                yerr=subset["roc_auc_std"],
                width=0.35,
                label=model,
                capsize=3,
            )
        ax.set_xticks(np.arange(len(subset)) + 0.35 / 2)
        ax.set_xticklabels(subset["scenario"], rotation=20, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Stress test: clinical/imaging missingness")
        ax.set_ylim(0, 1.0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "stress_test_roc_auc.png", dpi=300)
        fig.savefig(out_dir / "stress_test_roc_auc.pdf")
        plt.close(fig)
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    logger.info("Saved stress test summary to %s", out_dir / "stress_test_summary.csv")


if __name__ == "__main__":
    main()
