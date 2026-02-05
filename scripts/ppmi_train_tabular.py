import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pd_fusion.data.ppmi_studydata import create_splits
from pd_fusion.utils.metrics import compute_metrics


DEFAULT_MODELS = ["logreg", "lgbm", "mlp"]


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppmi_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)

        file_handler = logging.FileHandler(out_dir / "ppmi_train_tabular.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _select_feature_cols(schema: Dict, groups: List[str]) -> List[str]:
    cols: List[str] = []
    for group in groups:
        cols.extend(schema["groups"].get(group, {}).get("features", []))
    return cols


def _split_df(df: pd.DataFrame, split_ids: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["subject_id"].isin(split_ids["train"])].copy()
    val_df = df[df["subject_id"].isin(split_ids["val"])].copy()
    test_df = df[df["subject_id"].isin(split_ids["test"])].copy()
    return train_df, val_df, test_df


def _build_preprocessor(scale: bool, numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot),
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def _fit_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    return X_train_t, X_val_t, X_test_t


def _get_tree_model(seed: int, logger: logging.Logger):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            class_weight="balanced",
        )
    except Exception as exc:
        logger.warning("LightGBM not available: %s", exc)
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    except Exception as exc:
        logger.warning("XGBoost not available: %s", exc)

    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=seed)


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    hidden = cfg.get("hidden_dims", [128, 64])
    dropout = cfg.get("dropout", 0.3)
    max_epochs = cfg.get("max_epochs", 100)
    lr = cfg.get("lr", 1e-3)
    patience = cfg.get("patience", 10)

    layers = []
    prev = input_dim
    for dim in hidden:
        layers.append(torch.nn.Linear(prev, dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        prev = dim
    layers.append(torch.nn.Linear(prev, 1))
    model = torch.nn.Sequential(*layers).to(device)

    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_auc = -np.inf
    best_state = None
    patience_ctr = 0

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t).squeeze(1)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze(1)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        try:
            val_auc = roc_auc_score(y_val, val_prob)
        except ValueError:
            val_auc = 0.0
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(X_train_t).squeeze(1)).cpu().numpy()
        val_prob = torch.sigmoid(model(X_val_t).squeeze(1)).cpu().numpy()
    return train_prob, val_prob, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPMI tabular baselines")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"runs/ppmi_tabular_{timestamp}")
    logger = setup_logging(out_dir)
    import yaml
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    processed_dir = Path(cfg["processed_ppmi_dir"])
    level = cfg.get("modeling_level", "baseline")
    dataset_path = processed_dir / ("ppmi_visit_level.csv" if level == "visit" else "ppmi_subject_baseline.csv")
    schema_path = processed_dir / "ppmi_feature_schema.json"

    df = pd.read_csv(dataset_path, low_memory=False)
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)
    schema = json.loads(schema_path.read_text())

    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42)

    ablations = cfg.get("ablations", [])
    if not ablations:
        ablations = [
            {"name": "clinical_only", "groups": ["clinical"]},
            {"name": "mri_only", "groups": ["mri_derived"]},
            {"name": "datsbr_only", "groups": ["datsbr"]},
            {"name": "clinical_mri", "groups": ["clinical", "mri_derived"]},
            {"name": "clinical_datsbr", "groups": ["clinical", "datsbr"]},
            {"name": "full_fusion", "groups": ["clinical", "mri_derived", "datsbr", "nonmotor"]},
        ]

    models = cfg.get("models", DEFAULT_MODELS)

    split_cfg = cfg.get("splits", {})
    seeds = split_cfg.get("seeds", [42, 43, 44, 45, 46])
    if args.seed is not None:
        seeds = [args.seed]

    results = []

    for seed in seeds:
        split_path = processed_dir / f"ppmi_splits_seed{seed}.json"
        if split_path.exists():
            split_ids = json.loads(split_path.read_text())
            split_ids = {
                k: [str(v) for v in ids]
                for k, ids in split_ids.items()
            }
        else:
            labels = df.set_index("subject_id")["label"]
            split_ids = create_splits(labels, [seed], split_cfg)[seed]
        train_df, val_df, test_df = _split_df(df, split_ids)
        if train_df.empty or val_df.empty or test_df.empty:
            logger.warning(
                "Empty split for seed %s (train=%d, val=%d, test=%d); check subject_id types.",
                seed,
                len(train_df),
                len(val_df),
                len(test_df),
            )
            continue

        for ablation in ablations:
            feat_cols = _select_feature_cols(schema, ablation["groups"])
            feat_cols = [c for c in feat_cols if c in df.columns]
            if not feat_cols:
                logger.warning("No features found for ablation %s", ablation["name"])
                continue

            X_train = train_df[feat_cols]
            X_val = val_df[feat_cols]
            X_test = test_df[feat_cols]
            y_train = train_df["label"].to_numpy()
            y_val = val_df["label"].to_numpy()
            y_test = test_df["label"].to_numpy()

            numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
            cat_cols = [c for c in feat_cols if c not in numeric_cols]

            for model_name in models:
                if model_name == "logreg":
                    preprocessor = _build_preprocessor(True, numeric_cols, cat_cols)
                    X_train_t, X_val_t, X_test_t = _fit_transform(preprocessor, X_train, X_val, X_test)
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                    clf.fit(X_train_t, y_train)
                    y_prob = clf.predict_proba(X_test_t)[:, 1]
                elif model_name == "lgbm":
                    preprocessor = _build_preprocessor(False, numeric_cols, cat_cols)
                    X_train_t, X_val_t, X_test_t = _fit_transform(preprocessor, X_train, X_val, X_test)
                    clf = _get_tree_model(seed, logger)
                    clf.fit(X_train_t, y_train)
                    if hasattr(clf, "predict_proba"):
                        y_prob = clf.predict_proba(X_test_t)[:, 1]
                    else:
                        y_prob = clf.predict(X_test_t)
                elif model_name == "mlp":
                    preprocessor = _build_preprocessor(True, numeric_cols, cat_cols)
                    X_train_t, X_val_t, X_test_t = _fit_transform(preprocessor, X_train, X_val, X_test)
                    _, _, mlp_model = _train_mlp(X_train_t, y_train, X_val_t, y_val, seed, cfg.get("mlp", {}))
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    mlp_model = mlp_model.to(device)
                    with torch.no_grad():
                        logits = mlp_model(torch.tensor(X_test_t, dtype=torch.float32, device=device)).squeeze(1)
                        y_prob = torch.sigmoid(logits).cpu().numpy()
                else:
                    logger.warning("Unknown model %s", model_name)
                    continue

                try:
                    metrics = compute_metrics(y_test, y_prob)
                except ValueError as exc:
                    logger.warning("Metric computation failed for %s/%s/%s: %s", model_name, ablation["name"], seed, exc)
                    metrics = {
                        "roc_auc": float("nan"),
                        "pr_auc": float("nan"),
                        "balanced_accuracy": float("nan"),
                        "f1": float("nan"),
                        "brier_score": float("nan"),
                        "ece": float("nan"),
                    }
                result = {
                    "seed": seed,
                    "ablation": ablation["name"],
                    "model": model_name,
                    **metrics,
                }
                results.append(result)

                preds_path = out_dir / f"pred_{model_name}_{ablation['name']}_seed{seed}.csv"
                pd.DataFrame({
                    "subject_id": test_df["subject_id"].values,
                    "y_true": y_test,
                    "y_prob": y_prob,
                }).to_csv(preds_path, index=False)

    results_df = pd.DataFrame(results)
    results_path = out_dir / "results_all.csv"
    results_df.to_csv(results_path, index=False)

    summary = (
        results_df.groupby(["model", "ablation"]).agg(["mean", "std"]).reset_index()
    )
    # flatten columns
    summary.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary_path = out_dir / "summary_sweep_mean.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("Saved results to %s", results_path)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
