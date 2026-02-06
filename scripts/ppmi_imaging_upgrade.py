import argparse
import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pd_fusion.utils.metrics import compute_metrics


ID_COLS = {"subject_id", "visit_id", "visit_month", "date"}
DEFAULT_GLOBAL_EXCLUDE = [
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

DEFAULT_NONMOTOR = [
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

DEFAULT_DATSBR = [
    r"datscan",
    r"sbr",
    r"putamen",
    r"caudate",
    r"striat",
    r"asym",
]

DEFAULT_MRI = [
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
    logger = logging.getLogger("ppmi_imaging")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)
        file_handler = logging.FileHandler(out_dir / "ppmi_imaging_upgrade.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "subject_id" in df.columns:
        df["subject_id"] = df["subject_id"].astype(str)
    return df


def select_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce")


def filter_cols(cols: List[str], patterns: List[str]) -> List[str]:
    if not patterns:
        return cols
    return [c for c in cols if any(re.search(p, c, re.IGNORECASE) for p in patterns)]


def exclude_cols(cols: List[str], patterns: List[str]) -> List[str]:
    if not patterns:
        return cols
    return [c for c in cols if not any(re.search(p, c, re.IGNORECASE) for p in patterns)]


def get_feature_cols(df: pd.DataFrame, exclude_patterns: List[str]) -> List[str]:
    cols = [c for c in df.columns if c not in ID_COLS and c != "label"]
    cols = exclude_cols(cols, exclude_patterns)
    num_df = select_numeric(df, cols)
    return [c for c in num_df.columns if num_df[c].notna().any()]


def compute_missingness(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(columns=["feature", "missing_rate"])
    X = select_numeric(df, cols)
    missing = X.isna().mean().reset_index()
    missing.columns = ["feature", "missing_rate"]
    return missing.sort_values("missing_rate", ascending=False)


def missingness_by_subject(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(columns=["subject_id", "missing_rate"])
    X = select_numeric(df, cols)
    rates = X.isna().mean(axis=1)
    out = pd.DataFrame({"subject_id": df["subject_id"].astype(str), "missing_rate": rates})
    return out


def add_asymmetry_features(df: pd.DataFrame, dat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    new_cols = []
    paired = {}
    for col in dat_cols:
        if re.search(r"(_L_|_LEFT_|_L$|_LEFT$)", col, re.IGNORECASE):
            base = re.sub(r"(_L_|_LEFT_|_L$|_LEFT$)", "", col, flags=re.IGNORECASE)
            paired.setdefault(base, {})["L"] = col
        elif re.search(r"(_R_|_RIGHT_|_R$|_RIGHT$)", col, re.IGNORECASE):
            base = re.sub(r"(_R_|_RIGHT_|_R$|_RIGHT$)", "", col, flags=re.IGNORECASE)
            paired.setdefault(base, {})["R"] = col
    for base, sides in paired.items():
        if "L" in sides and "R" in sides:
            lcol = sides["L"]
            rcol = sides["R"]
            new_name = f"{base}_ASYM"
            lvals = pd.to_numeric(out[lcol], errors="coerce")
            rvals = pd.to_numeric(out[rcol], errors="coerce")
            out[new_name] = (lvals - rvals) / (lvals + rvals + 1e-6)
            new_cols.append(new_name)
    return out, new_cols


def build_covariate_matrix(
    df: pd.DataFrame,
    numeric_covs: List[str],
    categorical_covs: List[str],
    encoder: Optional[OneHotEncoder] = None,
) -> Tuple[np.ndarray, Optional[OneHotEncoder]]:
    num = pd.DataFrame()
    for col in numeric_covs:
        if col in df.columns:
            num[col] = pd.to_numeric(df[col], errors="coerce")
    if not num.empty:
        num = num.fillna(num.median())
    cat = pd.DataFrame()
    for col in categorical_covs:
        if col in df.columns:
            cat[col] = df[col].astype(str).fillna("UNKNOWN")
    if encoder is None:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_mat = encoder.fit_transform(cat) if not cat.empty else np.zeros((len(df), 0))
    else:
        cat_mat = encoder.transform(cat) if not cat.empty else np.zeros((len(df), 0))
    num_mat = num.to_numpy() if not num.empty else np.zeros((len(df), 0))
    X = np.concatenate([num_mat, cat_mat], axis=1)
    return X, encoder


def adjust_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    numeric_covs: List[str],
    categorical_covs: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not feature_cols:
        return train_df, test_df
    X_train = select_numeric(train_df, feature_cols)
    X_test = select_numeric(test_df, feature_cols)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    C_train, encoder = build_covariate_matrix(train_df, numeric_covs, categorical_covs, None)
    C_test, _ = build_covariate_matrix(test_df, numeric_covs, categorical_covs, encoder)
    if C_train.shape[1] == 0:
        return train_df, test_df
    reg = LinearRegression()
    reg.fit(C_train, X_train)
    train_adj = X_train - reg.predict(C_train)
    test_adj = X_test - reg.predict(C_test)
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in feature_cols:
        train_out[col] = train_adj[col].values
        test_out[col] = test_adj[col].values
    return train_out, test_out


def site_zscore_harmonize(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    site_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if site_col not in train_df.columns:
        return train_df, test_df
    train = train_df.copy()
    test = test_df.copy()
    X_train = select_numeric(train, feature_cols)
    X_test = select_numeric(test, feature_cols)
    global_mean = X_train.mean()
    global_std = X_train.std().replace(0, 1.0)
    for site, idx in train.groupby(site_col).groups.items():
        site_mean = X_train.loc[idx].mean()
        site_std = X_train.loc[idx].std().replace(0, 1.0)
        X_train.loc[idx] = (X_train.loc[idx] - site_mean) / site_std * global_std + global_mean
    if site_col in test.columns:
        for site, idx in test.groupby(site_col).groups.items():
            if site in train[site_col].unique():
                site_mean = X_train[train[site_col] == site].mean()
                site_std = X_train[train[site_col] == site].std().replace(0, 1.0)
            else:
                site_mean = global_mean
                site_std = global_std
            X_test.loc[idx] = (X_test.loc[idx] - site_mean) / site_std * global_std + global_mean
    for col in feature_cols:
        train[col] = X_train[col].values
        test[col] = X_test[col].values
    return train, test


def apply_harmonization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    method: str,
    site_cols: List[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if method == "none" or not feature_cols:
        return train_df, test_df
    if method == "combat":
        try:
            from neuroCombat import neuroCombat
            # Use first available site column
            site_col = next((c for c in site_cols if c in train_df.columns), None)
            if site_col is None:
                return train_df, test_df
            X_train = select_numeric(train_df, feature_cols).fillna(0).T
            covars = pd.DataFrame({"batch": train_df[site_col].astype(str)})
            combat = neuroCombat(dat=X_train, covars=covars, batch_col="batch")
            train_adj = pd.DataFrame(combat["data"].T, columns=feature_cols, index=train_df.index)
            test_adj = train_adj.reindex(test_df.index, fill_value=np.nan)
            train_out = train_df.copy()
            test_out = test_df.copy()
            for col in feature_cols:
                train_out[col] = train_adj[col].values
                test_out[col] = test_adj[col].values
            return train_out, test_out
        except Exception as exc:
            logger.warning("ComBat not available, falling back to site_zscore (%s)", exc)
            method = "site_zscore"
    if method == "site_zscore":
        site_col = next((c for c in site_cols if c in train_df.columns), None)
        if site_col is None:
            return train_df, test_df
        return site_zscore_harmonize(train_df, test_df, feature_cols, site_col)
    return train_df, test_df


def build_endpoint_labels(
    baseline_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    endpoint_cfg: Dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    endpoint = endpoint_cfg.get("type", "pd_vs_hc")
    horizon = endpoint_cfg.get("horizon_months", 24)
    if endpoint == "pd_vs_hc":
        return baseline_df

    visit_df = visit_df.copy()
    visit_df = visit_df.dropna(subset=["label"]).copy()
    if "visit_month" not in visit_df.columns:
        raise ValueError("visit_month required for longitudinal endpoints")
    # Derive visit_month from visit_id if missing
    if visit_df["visit_month"].isna().all():
        if "visit_id" in visit_df.columns:
            s = visit_df["visit_id"].astype(str).str.upper()
            derived = pd.to_numeric(s.str.extract(r"(\\d+)", expand=False), errors="coerce")
            bl_mask = s.isin({"BL", "BASELINE", "SCR", "SCREEN", "SC", "ENRL"})
            derived.loc[bl_mask] = 0
            visit_df = visit_df.copy()
            visit_df["visit_month"] = derived
            logger.info("Derived visit_month from visit_id for longitudinal endpoints")
        else:
            raise ValueError("visit_month missing and visit_id not available")

    base = baseline_df[["subject_id"]].copy()

    if endpoint.startswith("conversion"):
        # Only subjects who are HC at baseline can convert
        baseline_labels = baseline_df[["subject_id", "label"]].copy()
        base = base.merge(baseline_labels, on="subject_id", how="left")
        base = base[base["label"] == 0].copy()
        visit_df = visit_df[visit_df["subject_id"].isin(base["subject_id"])].copy()
        visit_df = visit_df[(visit_df["visit_month"].notna()) & (visit_df["visit_month"] <= horizon)]
        conv = visit_df.groupby("subject_id")["label"].max().rename("conversion_label")
        base = base.merge(conv, on="subject_id", how="left")
        base["conversion_label"] = base["conversion_label"].fillna(0).astype(int)
        base["label"] = base["conversion_label"]
        base = base.drop(columns=["conversion_label"])
        logger.info("Conversion endpoint: %d subjects", len(base))
        out = baseline_df.drop(columns=["label"], errors="ignore").merge(
            base[["subject_id", "label"]], on="subject_id", how="right"
        )
        return out

    if endpoint.startswith("progression"):
        feature = endpoint_cfg.get("progression_feature", "mds_updrs__NP3TOT")
        threshold = endpoint_cfg.get("progression_threshold", 5.0)
        allow_beyond = bool(endpoint_cfg.get("progression_allow_beyond_horizon", True))
        max_months = endpoint_cfg.get("progression_max_months", None)

        baseline_feature = baseline_df[["subject_id", feature]].copy()
        visit_df = visit_df.copy()
        visit_df = visit_df[visit_df[feature].notna()].copy()
        visit_df = visit_df[visit_df["visit_month"].notna()].copy()
        visit_df["visit_month"] = pd.to_numeric(visit_df["visit_month"], errors="coerce")
        visit_df = visit_df[visit_df["visit_month"].notna()].copy()

        if max_months is not None:
            visit_df = visit_df[visit_df["visit_month"] <= max_months].copy()

        # Prefer latest visit <= horizon
        target = visit_df[visit_df["visit_month"] <= horizon]
        target = target.sort_values("visit_month").groupby("subject_id").last()

        used_beyond = 0
        if allow_beyond:
            # For subjects with no visit <= horizon, use earliest visit > horizon
            future = visit_df[visit_df["visit_month"] > horizon]
            future = future.sort_values("visit_month").groupby("subject_id").first()
            missing_subjects = future.index.difference(target.index)
            if len(missing_subjects) > 0:
                target = pd.concat([target, future.loc[missing_subjects]], axis=0)
                used_beyond = len(missing_subjects)

        target = target.reset_index()
        if target.empty:
            raise ValueError(
                f"No progression targets found for feature {feature} (horizon={horizon}). "
                "Check visit_month/visit_id parsing or choose a different progression_feature."
            )

        if used_beyond > 0:
            logger.info("Progression: using %d subjects with visits beyond %s months", used_beyond, horizon)

        merged = baseline_feature.merge(
            target[["subject_id", feature]], on="subject_id", suffixes=("_base", "_target")
        )
        merged["delta"] = merged[f"{feature}_target"] - merged[f"{feature}_base"]
        merged["label"] = (merged["delta"] >= threshold).astype(int)
        out = baseline_df.drop(columns=["label"], errors="ignore").merge(
            merged[["subject_id", "label"]], on="subject_id", how="inner"
        )
        logger.info("Progression endpoint: %d subjects", len(out))
        return out

    raise ValueError(f"Unknown endpoint: {endpoint}")


def fit_model(model_name: str, seed: int, num_threads: int, logger: logging.Logger):
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=num_threads)
    if model_name == "lgbm":
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
            logger.warning("LightGBM not available (%s); using HistGradientBoosting", exc)
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown model {model_name}")


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
        clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=num_threads)
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        results.append({"repeat": i + 1, "roc_auc": auc})
    return results


def paired_auc_test(fold_df: pd.DataFrame, setting_a: str, setting_b: str) -> Dict:
    a = fold_df[(fold_df["setting"] == setting_a) & (fold_df["model"] == "lgbm")]["roc_auc"]
    b = fold_df[(fold_df["setting"] == setting_b) & (fold_df["model"] == "lgbm")]["roc_auc"]
    if len(a) != len(b) or len(a) == 0:
        return {"setting_a": setting_a, "setting_b": setting_b, "p_value": None}
    try:
        from scipy.stats import ttest_rel
        stat, pval = ttest_rel(a, b)
    except Exception:
        pval = None
    return {"setting_a": setting_a, "setting_b": setting_b, "p_value": pval}


def main():
    parser = argparse.ArgumentParser(description="PPMI imaging upgrade suite")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    endpoint_cfg = cfg.get("endpoint", {})
    seeds = cfg.get("cv", {}).get("seeds", [42])
    folds = int(cfg.get("cv", {}).get("folds", 5))
    cohort_cfg = cfg.get("cohort", {})

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"runs/ppmi_imaging_upgrade_{timestamp}")
    logger = setup_logging(out_dir)

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_threads)
    mpl_cache = out_dir / "mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    baseline_csv = Path(cfg["baseline_csv"])
    visit_csv = Path(cfg["visit_csv"])
    baseline_df = load_dataset(baseline_csv)
    visit_df = load_dataset(visit_csv) if visit_csv.exists() else None

    if visit_df is None:
        raise ValueError("visit_csv not found")

    df = build_endpoint_labels(baseline_df, visit_df, endpoint_cfg, logger)
    df = df.dropna(subset=["label"]).copy()

    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=seeds[0])

    exclude_patterns = cfg.get("feature_groups", {}).get("global_exclude_patterns", DEFAULT_GLOBAL_EXCLUDE)
    all_features = get_feature_cols(df, exclude_patterns)

    dat_patterns = cfg.get("feature_groups", {}).get("datsbr_patterns", DEFAULT_DATSBR)
    mri_patterns = cfg.get("feature_groups", {}).get("mri_patterns", DEFAULT_MRI)
    nonmotor_patterns = cfg.get("feature_groups", {}).get("non_motor_patterns", DEFAULT_NONMOTOR)

    dat_cols = filter_cols(all_features, dat_patterns)
    mri_cols = filter_cols(all_features, mri_patterns)
    nonmotor_cols = filter_cols(all_features, nonmotor_patterns)

    df, asym_cols = add_asymmetry_features(df, dat_cols)
    dat_cols = dat_cols + asym_cols

    imaging_cols = sorted(set(dat_cols + mri_cols))
    settings = {
        "non_motor_only": nonmotor_cols,
        "datsbr_only": dat_cols,
        "freesurfer_only": mri_cols,
        "fusion_nonmotor_imaging": sorted(set(nonmotor_cols + imaging_cols)),
    }

    kept_dropped = {}
    for name, cols in settings.items():
        kept_dropped[name] = {"kept": cols, "dropped": [c for c in all_features if c not in cols]}

    (out_dir / "kept_dropped_columns.json").write_text(json.dumps(kept_dropped, indent=2))
    (out_dir / "imaging_columns.json").write_text(json.dumps({"datsbr": dat_cols, "mri": mri_cols}, indent=2))

    # Imaging-available cohort filtering
    def _availability_mask(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        dat_mask = (
            select_numeric(frame, dat_cols).notna().any(axis=1).to_numpy()
            if dat_cols else np.zeros(len(frame), dtype=bool)
        )
        mri_mask = (
            select_numeric(frame, mri_cols).notna().any(axis=1).to_numpy()
            if mri_cols else np.zeros(len(frame), dtype=bool)
        )
        return dat_mask, mri_mask

    dat_avail, mri_avail = _availability_mask(df)
    any_imaging = dat_avail | mri_avail
    avail_summary = {
        "total_subjects": int(len(df)),
        "dat_available": int(dat_avail.sum()),
        "mri_available": int(mri_avail.sum()),
        "any_imaging_available": int(any_imaging.sum()),
        "dat_available_rate": float(dat_avail.mean()) if len(df) else 0.0,
        "mri_available_rate": float(mri_avail.mean()) if len(df) else 0.0,
        "any_imaging_available_rate": float(any_imaging.mean()) if len(df) else 0.0,
    }
    (out_dir / "imaging_availability_summary.json").write_text(json.dumps(avail_summary, indent=2))

    if cohort_cfg.get("imaging_available_only", False):
        require_dat = cohort_cfg.get("require_dat", False)
        require_mri = cohort_cfg.get("require_mri", False)
        require_any = cohort_cfg.get("require_any", True)
        if require_dat and require_mri:
            mask = dat_avail & mri_avail
        elif require_dat:
            mask = dat_avail
        elif require_mri:
            mask = mri_avail
        elif require_any:
            mask = any_imaging
        else:
            mask = np.ones(len(df), dtype=bool)
        df = df.loc[mask].copy()
        logger.info("Imaging-available cohort filter applied: %d subjects", len(df))

    # Audit missingness
    miss_feat = compute_missingness(df, imaging_cols)
    miss_feat.to_csv(out_dir / "imaging_missingness_per_feature.csv", index=False)
    miss_subj = missingness_by_subject(df, imaging_cols)
    miss_subj.to_csv(out_dir / "imaging_missingness_per_subject.csv", index=False)

    cov_cfg = cfg.get("covariates", {})
    num_covs = cov_cfg.get("numeric", [])
    cat_covs = cov_cfg.get("categorical", [])
    (out_dir / "covariates_used.json").write_text(json.dumps({"numeric": num_covs, "categorical": cat_covs}, indent=2))

    harm_cfg = cfg.get("harmonization", {})
    harm_method = harm_cfg.get("method", "none")
    harm_site_cols = harm_cfg.get("site_cols", [])

    per_fold_rows = []
    feature_rows = []
    univariate_rows = []
    pred_rows = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for setting, feature_cols in settings.items():
            if not feature_cols:
                logger.warning("No features for %s", setting)
                continue
            uni = compute_univariate_auc(df, df["label"].values, feature_cols)
            for feat, auc in uni:
                univariate_rows.append({"setting": setting, "feature": feat, "auc": auc, "seed": seed})

            for fold, (train_idx, test_idx) in enumerate(skf.split(df, df["label"].values), start=1):
                train_df = df.iloc[train_idx].copy()
                test_df = df.iloc[test_idx].copy()

                imaging_in_setting = [c for c in feature_cols if c in imaging_cols]
                if imaging_in_setting:
                    train_df, test_df = adjust_features(train_df, test_df, imaging_in_setting, num_covs, cat_covs)
                    train_df, test_df = apply_harmonization(
                        train_df, test_df, imaging_in_setting, harm_method, harm_site_cols, logger
                    )

                X_train = select_numeric(train_df, feature_cols)
                X_test = select_numeric(test_df, feature_cols)

                imputer = SimpleImputer(strategy="median", add_indicator=True)
                X_train_imp = imputer.fit_transform(X_train)
                X_test_imp = imputer.transform(X_test)

                feat_names = list(feature_cols)
                if imputer.indicator_ is not None:
                    for idx in imputer.indicator_.features_:
                        feat_names.append(f"{feature_cols[idx]}_missing")

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imp)
                X_test_scaled = scaler.transform(X_test_imp)

                for model_name in cfg.get("models", ["logreg", "lgbm"]):
                    clf = fit_model(model_name, seed + fold, args.num_threads, logger)
                    Xtr = X_train_scaled if model_name == "logreg" else X_train_imp
                    Xte = X_test_scaled if model_name == "logreg" else X_test_imp
                    clf.fit(Xtr, train_df["label"].values)
                    y_prob = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xte)
                    metrics = compute_metrics(test_df["label"].values, y_prob)
                    per_fold_rows.append({
                        "seed": seed,
                        "fold": fold,
                        "setting": setting,
                        "model": model_name,
                        **metrics,
                    })
                    for idx, prob in zip(test_df.index, y_prob):
                        pred_rows.append({
                            "index": int(idx),
                            "subject_id": test_df.loc[idx, "subject_id"],
                            "setting": setting,
                            "model": model_name,
                            "fold": fold,
                            "seed": seed,
                            "y_true": int(test_df.loc[idx, "label"]),
                            "y_prob": float(prob),
                        })

                    if model_name == "logreg" and hasattr(clf, "coef_"):
                        imp = np.abs(clf.coef_.reshape(-1))
                    elif hasattr(clf, "feature_importances_"):
                        imp = clf.feature_importances_.astype(float)
                    else:
                        imp = None

                    if imp is not None:
                        for name, val in zip(feat_names, imp):
                            feature_rows.append({
                                "setting": setting,
                                "model": model_name,
                                "fold": fold,
                                "seed": seed,
                                "feature": name,
                                "importance": float(val),
                            })

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(out_dir / "per_fold_metrics.csv", index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    summary = per_fold_df.groupby(["setting", "model"]).agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(out_dir / "summary_mean.csv", index=False)

    feat_df = pd.DataFrame(feature_rows)
    if not feat_df.empty:
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
    for setting in ["non_motor_only", "fusion_nonmotor_imaging"]:
        cols = settings.get(setting, [])
        if not cols:
            continue
        results = permutation_test(df, cols, args.num_threads, repeats=5)
        for row in results:
            row["setting"] = setting
            perm_rows.append(row)
    pd.DataFrame(perm_rows).to_csv(out_dir / "permutation_test.csv", index=False)

    paired = paired_auc_test(per_fold_df, "non_motor_only", "fusion_nonmotor_imaging")
    (out_dir / "paired_tests.json").write_text(json.dumps(paired, indent=2))

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            # ROC bar (best model per setting)
            plot_df = summary.copy().sort_values("roc_auc_mean", ascending=False)
            best_rows = []
            for setting in plot_df["setting"].unique():
                subset = plot_df[plot_df["setting"] == setting]
                best_rows.append(subset.iloc[0])
            best_df = pd.DataFrame(best_rows)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(best_df["setting"], best_df["roc_auc_mean"], yerr=best_df["roc_auc_std"], capsize=4)
            ax.set_ylabel("ROC-AUC")
            ax.set_title("PPMI Imaging Upgrade: ROC-AUC")
            ax.set_ylim(0, 1.0)
            plt.xticks(rotation=25, ha="right")
            fig.tight_layout()
            fig.savefig(out_dir / "roc_auc_bar.png", dpi=200)
            plt.close(fig)

            # ROC curves for key settings
            fig, ax = plt.subplots(figsize=(8, 6))
            key_settings = ["non_motor_only", "fusion_nonmotor_imaging"]
            for setting in key_settings:
                sub = pred_df[(pred_df["setting"] == setting) & (pred_df["model"] == "lgbm")]
                if sub.empty:
                    continue
                fpr, tpr, _ = roc_curve(sub["y_true"], sub["y_prob"])
                auc = roc_auc_score(sub["y_true"], sub["y_prob"])
                ax.plot(fpr, tpr, label=f"{setting} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_title("ROC Curves (LGBM)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "roc_curves.png", dpi=200)
            plt.close(fig)

            # Calibration curves for key settings
            fig, ax = plt.subplots(figsize=(6, 5))
            for setting in key_settings:
                sub = pred_df[(pred_df["setting"] == setting) & (pred_df["model"] == "lgbm")]
                if sub.empty:
                    continue
                frac_pos, mean_pred = calibration_curve(sub["y_true"], sub["y_prob"], n_bins=10)
                ax.plot(mean_pred, frac_pos, marker="o", label=setting)
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set_title("Calibration Curves (LGBM)")
            ax.set_xlabel("Mean predicted")
            ax.set_ylabel("Fraction positive")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "calibration_curves.png", dpi=200)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Plot generation failed: %s", exc)

    if not args.no_shap:
        try:
            import shap
            best = summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
            setting = best["setting"]
            model = best["model"]
            feature_cols = settings.get(setting, [])
            if feature_cols:
                full_df = df.copy()
                imaging_in_setting = [c for c in feature_cols if c in imaging_cols]
                if imaging_in_setting:
                    full_df, _ = adjust_features(full_df, full_df, imaging_in_setting, num_covs, cat_covs)
                    full_df, _ = apply_harmonization(full_df, full_df, imaging_in_setting, harm_method, harm_site_cols, logger)
                X_full = select_numeric(full_df, feature_cols)
                imputer = SimpleImputer(strategy="median", add_indicator=True)
                X_imp = imputer.fit_transform(X_full)
                feat_names = list(feature_cols)
                if imputer.indicator_ is not None:
                    for idx in imputer.indicator_.features_:
                        feat_names.append(f"{feature_cols[idx]}_missing")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_imp)
                clf = fit_model(model, seeds[0], args.num_threads, logger)
                X_train = X_scaled if model == "logreg" else X_imp
                clf.fit(X_train, full_df["label"].values)
                sample_idx = np.random.default_rng(seeds[0]).choice(len(full_df), size=min(500, len(full_df)), replace=False)
                X_sample = X_train[sample_idx]
                if model == "lgbm" and hasattr(clf, "predict_proba"):
                    explainer = shap.TreeExplainer(clf)
                    shap_vals = explainer.shap_values(X_sample)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                else:
                    explainer = shap.LinearExplainer(clf, X_sample)
                    shap_vals = explainer.shap_values(X_sample)
                mean_abs = np.mean(np.abs(shap_vals), axis=0)
                shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
                shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
                shap_df.to_csv(out_dir / "shap_summary.csv", index=False)
        except Exception as exc:
            logger.warning("SHAP computation skipped: %s", exc)

    logger.info("Saved summary to %s", out_dir / "summary_mean.csv")


if __name__ == "__main__":
    main()
