import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from pd_fusion.utils.metrics import compute_metrics


ID_COLS = {"subject_id", "visit_id", "visit_month", "date"}

SETTINGS = {
    "full_clinical": {
        "type": "all_numeric",
    },
    "no_motor_exam": {
        "type": "drop_regex",
        "drop_regex": [
            r"^mds_updrs__.*",
            r".*NHY.*",
            r".*TRMR.*",
            r".*RIG.*",
            r".*BRADY.*",
        ],
    },
    "non_motor_only": {
        "type": "allow_regex",
        "allow_regex": [
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
        ],
    },
    "datsbr_only": {
        "type": "allow_regex",
        "allow_regex": [
            r"datscan",
            r"sbr",
            r"putamen",
            r"caudate",
            r"striat",
            r"dat",
        ],
    },
    "freesurfer_only": {
        "type": "allow_regex",
        "allow_regex": [
            r"mri_derived__",
            r"thickness",
            r"cortical",
            r"volume",
            r"area",
            r"aseg",
            r"hippo",
            r"entorhinal",
            r"amygdala",
        ],
    },
    "fusion_nonmotor_imaging": {
        "type": "union",
        "sources": ["non_motor_only", "datsbr_only", "freesurfer_only"],
    },
}

MODELS = ["logreg", "lgbm"]


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppmi_suite")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(out_dir / "ppmi_meaningful_suite.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)
    return df


def select_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def get_all_numeric_features(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in ID_COLS and c != "label"]
    num_df = select_numeric(df, cols)
    keep = [c for c in num_df.columns if num_df[c].notna().any()]
    return keep


def apply_setting(df: pd.DataFrame, setting: str, cache: Dict[str, List[str]]) -> List[str]:
    if setting in cache:
        return cache[setting]
    spec = SETTINGS[setting]
    if spec["type"] == "all_numeric":
        cols = get_all_numeric_features(df)
    elif spec["type"] == "drop_regex":
        base = get_all_numeric_features(df)
        cols = base.copy()
        for pattern in spec["drop_regex"]:
            cols = [c for c in cols if not re.search(pattern, c, re.IGNORECASE)]
    elif spec["type"] == "allow_regex":
        base = get_all_numeric_features(df)
        cols = [
            c for c in base
            if any(re.search(p, c, re.IGNORECASE) for p in spec["allow_regex"])
        ]
    elif spec["type"] == "union":
        cols = []
        for src in spec["sources"]:
            cols.extend(apply_setting(df, src, cache))
        cols = sorted(set(cols))
    else:
        cols = []
    cache[setting] = cols
    return cols


def prepare_matrices(df: pd.DataFrame, feature_cols: List[str], scale: bool):
    X = select_numeric(df, feature_cols)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    X_imp = imputer.fit_transform(X)
    feature_names = list(feature_cols)
    if imputer.indicator_ is not None:
        missing_idx = imputer.indicator_.features_
        for idx in missing_idx:
            feature_names.append(f"{feature_cols[idx]}_missing")
    if scale:
        scaler = StandardScaler()
        X_imp = scaler.fit_transform(X_imp)
    else:
        scaler = None
    return X_imp, imputer, scaler, feature_names


def transform_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    imputer: SimpleImputer,
    scaler: Optional[StandardScaler],
):
    X = select_numeric(df, feature_cols)
    X_imp = imputer.transform(X)
    if scaler is not None:
        X_imp = scaler.transform(X_imp)
    return X_imp


def get_lgbm(seed: int, num_threads: int, logger: logging.Logger):
    try:
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
    except Exception as exc:
        logger.warning("LightGBM not available (%s); falling back to HistGradientBoosting", exc)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed)


def compute_univariate_auc(df: pd.DataFrame, y: np.ndarray, feature_cols: List[str], top_k: int = 20):
    scores = []
    X = select_numeric(df, feature_cols)
    for col in feature_cols:
        x = X[col].fillna(X[col].median())
        try:
            auc = roc_auc_score(y, x)
            scores.append((col, auc))
        except Exception:
            continue
    scores = sorted(scores, key=lambda x: abs(x[1] - 0.5), reverse=True)[:top_k]
    return scores


def permutation_test(df: pd.DataFrame, feature_cols: List[str], num_threads: int, repeats: int = 5):
    X = select_numeric(df, feature_cols).fillna(0)
    y = df["label"].values
    results = []
    for i in range(repeats):
        y_perm = np.random.permutation(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_perm, test_size=0.2, random_state=42 + i, stratify=y_perm
        )
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=num_threads)
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        results.append({"repeat": i + 1, "roc_auc": auc})
    return results


def main():
    parser = argparse.ArgumentParser(description="PPMI meaningful baseline suite")
    parser.add_argument(
        "--input-csv",
        default="/home/adixit1/IEEE-spid/data/processed/ppmi/ppmi_subject_baseline.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"runs/ppmi_meaningful_suite_{timestamp}")
    logger = setup_logging(out_dir)

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_threads)
    mpl_cache = out_dir / "mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    df = load_dataset(Path(args.input_csv))
    df = df.dropna(subset=["label"]).copy()

    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)

    kept_dropped: Dict[str, Dict[str, List[str]]] = {}
    cache: Dict[str, List[str]] = {}

    for setting in SETTINGS:
        cols = apply_setting(df, setting, cache)
        if setting == "full_clinical":
            dropped = []
        elif SETTINGS[setting]["type"] == "drop_regex":
            dropped = [c for c in cache["full_clinical"] if c not in cols]
        else:
            dropped = [c for c in cache["full_clinical"] if c not in cols]
        kept_dropped[setting] = {"kept": cols, "dropped": dropped}

    (out_dir / "kept_dropped_columns.json").write_text(json.dumps(kept_dropped, indent=2))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    per_fold_rows = []
    feature_rows = []
    univariate_rows = []

    for setting in SETTINGS:
        feature_cols = kept_dropped[setting]["kept"]
        if not feature_cols:
            logger.warning("No features for setting %s", setting)
            continue

        # Univariate list
        uni = compute_univariate_auc(df, df["label"].values, feature_cols)
        for feat, auc in uni:
            univariate_rows.append({"setting": setting, "feature": feat, "auc": auc})

        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["label"].values), start=1):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            y_train = train_df["label"].values
            y_test = test_df["label"].values

            X_train, imputer, scaler, feat_names = prepare_matrices(
                train_df, feature_cols, scale=True
            )
            X_test = transform_matrix(test_df, feature_cols, imputer, scaler)

            for model_name in MODELS:
                if model_name == "logreg":
                    clf = LogisticRegression(
                        max_iter=2000, class_weight="balanced", n_jobs=args.num_threads
                    )
                else:
                    clf = get_lgbm(args.seed + fold, args.num_threads, logger)

                clf.fit(X_train, y_train)
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_test)[:, 1]
                else:
                    y_prob = clf.predict(X_test)

                metrics = compute_metrics(y_test, y_prob)
                per_fold_rows.append({
                    "setting": setting,
                    "model": model_name,
                    "fold": fold,
                    **metrics,
                })

                # feature importance
                if model_name == "logreg" and hasattr(clf, "coef_"):
                    coef = clf.coef_.reshape(-1)
                    imp = np.abs(coef)
                elif hasattr(clf, "feature_importances_"):
                    imp = clf.feature_importances_.astype(float)
                else:
                    logger.warning("No feature importances available for %s/%s", setting, model_name)
                    imp = None

                if imp is not None:
                    for name, val in zip(feat_names, imp):
                        feature_rows.append({
                            "setting": setting,
                            "model": model_name,
                            "fold": fold,
                            "feature": name,
                            "importance": float(val),
                        })

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(out_dir / "per_fold_metrics.csv", index=False)

    summary = per_fold_df.groupby(["setting", "model"]).agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out_dir / "summary_mean.csv", index=False)

    feat_df = pd.DataFrame(feature_rows)
    feat_summary = (
        feat_df.groupby(["setting", "model", "feature"])["importance"]
        .mean()
        .reset_index()
    )
    feat_summary = feat_summary.sort_values(["setting", "model", "importance"], ascending=[True, True, False])
    top_feats = feat_summary.groupby(["setting", "model"]).head(20)
    top_feats.to_csv(out_dir / "feature_importance.csv", index=False)

    pd.DataFrame(univariate_rows).to_csv(out_dir / "univariate_top.csv", index=False)

    perm_rows = []
    for setting in ["full_clinical", "fusion_nonmotor_imaging"]:
        cols = kept_dropped[setting]["kept"]
        if not cols:
            continue
        results = permutation_test(df, cols, args.num_threads, repeats=5)
        for row in results:
            row["setting"] = setting
            perm_rows.append(row)
    pd.DataFrame(perm_rows).to_csv(out_dir / "permutation_test.csv", index=False)

    # Plot
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            plot_df = summary.copy()
            plot_df = plot_df.sort_values("roc_auc_mean", ascending=False)
            best_rows = []
            for setting in plot_df["setting"].unique():
                subset = plot_df[plot_df["setting"] == setting]
                best_rows.append(subset.iloc[0])
            best_df = pd.DataFrame(best_rows)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(best_df["setting"], best_df["roc_auc_mean"], yerr=best_df["roc_auc_std"], capsize=4)
            ax.set_ylabel("ROC-AUC")
            ax.set_title("PPMI meaningful baselines")
            ax.set_ylim(0, 1.0)
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / "roc_auc_bar.png", dpi=200)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Plot generation failed: %s", exc)

    logger.info("Saved summary to %s", out_dir / "summary_mean.csv")


if __name__ == "__main__":
    main()
