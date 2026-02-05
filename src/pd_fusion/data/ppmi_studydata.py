import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SUBJECT_COLS = [
    "PATNO",
    "SUBJECT_ID",
    "SUBJECT",
    "PARTICIPANT_ID",
    "RID",
    "ID",
    "participant_id",
    "subject_id",
]
DEFAULT_VISIT_COLS = [
    "EVENT_ID",
    "VISIT_ID",
    "VISIT",
    "VISITID",
    "EVENT",
    "TIMEPOINT",
    "VISITNUM",
]
DEFAULT_VISIT_MONTH_COLS = [
    "VISIT_MONTH",
    "MONTH",
    "VISITMNTH",
    "MONTHS",
    "MONTHS_SINCE_BL",
    "MONTHS_SINCE_BASELINE",
]
DEFAULT_DATE_COLS = [
    "INFODT",
    "EXAMDATE",
    "EXAM_DATE",
    "DATE",
    "VISIT_DATE",
]

DEFAULT_LABEL_COLS = [
    "DIAGNOSIS",
    "COHORT",
    "COHORT_DESCRIPTION",
    "ENROLL_CAT",
    "CURRENT_DIAGNOSIS",
    "PRIMDIAG",
    "DX",
]

DEFAULT_POSITIVE_KEYS = ["pd", "parkinson", "parkinson's disease"]
DEFAULT_NEGATIVE_KEYS = ["hc", "healthy", "control"]
DEFAULT_EXCLUDE_KEYS = ["swedd", "prodromal", "genetic", "other", "unknown"]


@dataclass
class TableBundle:
    name: str
    group: str
    df: pd.DataFrame
    feature_cols: List[str]
    has_visit: bool
    raw_df: Optional[pd.DataFrame] = None


def _norm_col(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def _match_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    norm_map = {_norm_col(col): col for col in columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _detect_subject_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return _match_column(df.columns, candidates + DEFAULT_SUBJECT_COLS)


def _detect_visit_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return _match_column(df.columns, candidates + DEFAULT_VISIT_COLS)


def _detect_visit_month_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return _match_column(df.columns, candidates + DEFAULT_VISIT_MONTH_COLS)


def _detect_date_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return _match_column(df.columns, candidates + DEFAULT_DATE_COLS)


def _coerce_visit_month(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    vals = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(vals, errors="coerce")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _extract_zips(raw_dir: Path, logger: logging.Logger) -> None:
    zips = list(raw_dir.glob("**/*.zip"))
    if not zips:
        return
    extract_dir = raw_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    for zpath in zips:
        try:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(extract_dir)
                logger.info("Extracted %s into %s", zpath, extract_dir)
        except zipfile.BadZipFile:
            logger.warning("Skipping invalid zip: %s", zpath)


def _resolve_table_paths(study_dir: Path, patterns: List[str]) -> List[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend([Path(p) for p in study_dir.glob(pattern)])
        matches.extend([Path(p) for p in study_dir.glob(f"**/{pattern}")])
    # Deduplicate and keep existing files
    uniq = []
    seen = set()
    for p in matches:
        if p.exists() and p.is_file() and p.suffix.lower() in {".csv"}:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
    return uniq


def _canonicalize_table(
    df: pd.DataFrame,
    table_name: str,
    group: str,
    config: Dict,
) -> TableBundle:
    col_cfg = config.get("column_candidates", {})
    subj_col = _detect_subject_col(df, col_cfg.get("subject_id", []))
    if subj_col is None:
        raise ValueError(f"No subject id column found for {table_name}")
    visit_col = _detect_visit_col(df, col_cfg.get("visit_id", []))
    month_col = _detect_visit_month_col(df, col_cfg.get("visit_month", []))
    date_col = _detect_date_col(df, col_cfg.get("date", []))

    df = df.copy()
    df = df.rename(columns={subj_col: "subject_id"})
    df["subject_id"] = df["subject_id"].astype(str)

    has_visit = visit_col is not None
    if visit_col is not None:
        df = df.rename(columns={visit_col: "visit_id"})
        df["visit_id"] = df["visit_id"].astype(str)
    else:
        df["visit_id"] = pd.NA

    if month_col is not None:
        df = df.rename(columns={month_col: "visit_month"})
        df["visit_month"] = _coerce_visit_month(df["visit_month"])
    else:
        df["visit_month"] = pd.NA

    if date_col is not None:
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    if visit_col is not None:
        df = df.groupby(["subject_id", "visit_id"], as_index=False).first()
    else:
        df = df.groupby(["subject_id"], as_index=False).first()

    base_cols = ["subject_id", "visit_id", "visit_month", "date"]
    feature_cols = [c for c in df.columns if c not in set(base_cols)]

    if group == "labels":
        return TableBundle(
            name=table_name,
            group=group,
            df=df[base_cols + feature_cols],
            feature_cols=[],
            has_visit=has_visit,
            raw_df=df,
        )

    prefixed = {c: f"{table_name}__{c}" for c in feature_cols}
    df = df.rename(columns=prefixed)
    return TableBundle(
        name=table_name,
        group=group,
        df=df[base_cols + list(prefixed.values())],
        feature_cols=list(prefixed.values()),
        has_visit=has_visit,
        raw_df=None,
    )


def load_tables(config: Dict, logger: logging.Logger) -> List[TableBundle]:
    raw_dir = Path(config["study_data_dir"])
    if config.get("extract_zips", True):
        _extract_zips(raw_dir, logger)
    table_cfg = config.get("tables", {})

    bundles: List[TableBundle] = []
    for table_name, meta in table_cfg.items():
        patterns = meta.get("patterns", [])
        if not patterns:
            continue
        paths = _resolve_table_paths(raw_dir, patterns)
        if not paths:
            logger.warning("No files matched for %s", table_name)
            continue
        dfs = []
        for path in paths:
            try:
                df = _read_csv(path)
                dfs.append(df)
                logger.info("Loaded %s (%s)", path.name, table_name)
            except Exception as exc:
                logger.warning("Failed reading %s: %s", path, exc)
        if not dfs:
            continue
        df_all = pd.concat(dfs, ignore_index=True)
        try:
            bundle = _canonicalize_table(df_all, table_name, meta.get("group", "clinical"), config)
            bundles.append(bundle)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", table_name, exc)
    return bundles


def _normalize_label_value(value: object, config: Dict) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value_map = config.get("label", {}).get("value_map", {})
    if value in value_map:
        return value_map[value]
    s = str(value).strip().lower()
    if s in value_map:
        return value_map[s]
    for key in config.get("label", {}).get("exclude_values", DEFAULT_EXCLUDE_KEYS):
        if key in s:
            return None
    for key in config.get("label", {}).get("positive_values", DEFAULT_POSITIVE_KEYS):
        if key in s:
            return 1
    for key in config.get("label", {}).get("negative_values", DEFAULT_NEGATIVE_KEYS):
        if key in s:
            return 0
    return None


def infer_labels(label_tables: List[TableBundle], config: Dict, logger: logging.Logger) -> pd.DataFrame:
    diag_candidates = config.get("label", {}).get("diagnosis_column_candidates", DEFAULT_LABEL_COLS)
    labels: Dict[str, int] = {}
    conflicts = 0
    excluded = 0

    for bundle in label_tables:
        df = bundle.raw_df if bundle.raw_df is not None else bundle.df
        diag_col = _match_column(df.columns, diag_candidates)
        if diag_col is None:
            logger.warning("No diagnosis column found in %s", bundle.name)
            continue
        for _, row in df.iterrows():
            subject_id = str(row.get("subject_id"))
            label = _normalize_label_value(row.get(diag_col), config)
            if label is None:
                excluded += 1
                continue
            if subject_id not in labels:
                labels[subject_id] = label
            elif labels[subject_id] != label:
                conflicts += 1
    if conflicts:
        logger.warning("Conflicting labels found for %d subjects", conflicts)
    if excluded:
        logger.info("Excluded %d rows with non PD/HC labels", excluded)

    out = pd.DataFrame({"subject_id": list(labels.keys()), "label": list(labels.values())})
    return out


def _build_visits_df(tables: List[TableBundle]) -> pd.DataFrame:
    visit_tables = [t for t in tables if t.has_visit]
    if not visit_tables:
        subjects = sorted({sid for t in tables for sid in t.df["subject_id"].unique()})
        return pd.DataFrame({"subject_id": subjects, "visit_id": "BL", "visit_month": pd.NA, "date": pd.NaT})

    rows = []
    for table in visit_tables:
        subset = table.df[["subject_id", "visit_id", "visit_month", "date"]].copy()
        rows.append(subset)
    visits = pd.concat(rows, ignore_index=True).drop_duplicates()
    visits = visits.sort_values(["subject_id", "visit_month", "date"], na_position="last")
    return visits


def build_visit_level_dataset(tables: List[TableBundle], labels: pd.DataFrame) -> pd.DataFrame:
    visits = _build_visits_df(tables)
    merged = visits.copy()

    for table in tables:
        if table.group == "labels":
            continue
        if table.has_visit:
            table_df = table.df.copy()
            merged = merged.merge(
                table_df,
                on=["subject_id", "visit_id"],
                how="left",
                suffixes=("", f"_{table.name}"),
            )
        else:
            subject_df = table.df.drop(columns=["visit_id", "visit_month", "date"], errors="ignore")
            merged = merged.merge(subject_df, on="subject_id", how="left")
    merged = merged.merge(labels, on="subject_id", how="left")
    return merged


def select_baseline(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    baseline_cfg = config.get("baseline", {})
    priority = [str(p) for p in baseline_cfg.get("visit_id_priority", ["BL", "SC", "V01"])]

    def pick_group(group: pd.DataFrame) -> pd.Series:
        group = group.copy()
        group["visit_id"] = group["visit_id"].astype(str)
        for vid in priority:
            cand = group[group["visit_id"].str.upper() == vid.upper()]
            if not cand.empty:
                return cand.sort_values(["visit_month", "date"], na_position="last").iloc[0]
        return group.sort_values(["visit_month", "date"], na_position="last").iloc[0]

    baseline = df.groupby("subject_id", group_keys=False).apply(pick_group)
    baseline = baseline.reset_index(drop=True)
    return baseline


def build_feature_schema(df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict:
    schema = {"groups": {}, "feature_types": {}}
    for group, cols in feature_groups.items():
        missing = df[cols].isna().mean().to_dict() if cols else {}
        schema["groups"][group] = {
            "features": cols,
            "missing_rate": missing,
        }
        for col in cols:
            if col not in schema["feature_types"]:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    schema["feature_types"][col] = "numeric"
                else:
                    schema["feature_types"][col] = "categorical"
    return schema


def create_splits(labels: pd.Series, seeds: List[int], split_cfg: Dict) -> Dict[int, Dict[str, List[str]]]:
    splits: Dict[int, Dict[str, List[str]]] = {}
    subjects = labels.index.to_numpy()
    y = labels.to_numpy()
    train_size = split_cfg.get("train_size", 0.7)
    val_size = split_cfg.get("val_size", 0.15)
    test_size = split_cfg.get("test_size", 0.15)
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train/val/test sizes must sum to 1.0")

    for seed in seeds:
        train_ids, temp_ids, y_train, y_temp = train_test_split(
            subjects,
            y,
            train_size=train_size,
            stratify=y,
            random_state=seed,
        )
        val_ratio = val_size / (val_size + test_size)
        val_ids, test_ids = train_test_split(
            temp_ids,
            train_size=val_ratio,
            stratify=y_temp,
            random_state=seed,
        )
        splits[seed] = {
            "train": train_ids.tolist(),
            "val": val_ids.tolist(),
            "test": test_ids.tolist(),
        }
    return splits


def build_ppmi_datasets(config: Dict, logger: logging.Logger) -> Dict[str, Path]:
    processed_dir = Path(config["processed_ppmi_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_tables(config, logger)
    label_tables = [b for b in bundles if b.group == "labels"]
    labels = infer_labels(label_tables, config, logger)
    if labels.empty:
        raise ValueError("No PD/HC labels could be inferred from label tables.")

    feature_groups: Dict[str, List[str]] = {}
    for bundle in bundles:
        if bundle.group in {"labels"}:
            continue
        feature_groups.setdefault(bundle.group, []).extend(bundle.feature_cols)

    visit_df = build_visit_level_dataset(bundles, labels)
    baseline_df = select_baseline(visit_df, config)

    # Filter to PD/HC labels only
    baseline_df = baseline_df[baseline_df["label"].isin([0, 1])].copy()
    visit_df = visit_df[visit_df["label"].isin([0, 1])].copy()

    baseline_path = processed_dir / "ppmi_subject_baseline.csv"
    visit_path = processed_dir / "ppmi_visit_level.csv"

    baseline_df.to_csv(baseline_path, index=False)
    visit_df.to_csv(visit_path, index=False)

    schema = build_feature_schema(baseline_df, feature_groups)
    schema_path = processed_dir / "ppmi_feature_schema.json"
    schema["n_subjects"] = int(baseline_df["subject_id"].nunique())
    schema["n_visits"] = int(visit_df.shape[0])
    schema_path.write_text(json.dumps(schema, indent=2))

    split_cfg = config.get("splits", {})
    seeds = split_cfg.get("seeds", [42, 43, 44, 45, 46])
    labels_series = baseline_df.set_index("subject_id")["label"]
    splits = create_splits(labels_series, seeds, split_cfg)

    for seed, split in splits.items():
        split_path = processed_dir / f"ppmi_splits_seed{seed}.json"
        split_path.write_text(json.dumps(split, indent=2))

    manifest_path = processed_dir / "ppmi_manifest.md"
    _write_manifest(manifest_path, baseline_df, visit_df, feature_groups)

    return {
        "baseline": baseline_path,
        "visit_level": visit_path,
        "schema": schema_path,
        "manifest": manifest_path,
    }


def _write_manifest(path: Path, baseline_df: pd.DataFrame, visit_df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> None:
    lines = []
    lines.append("# PPMI Study Data Manifest")
    lines.append("")
    lines.append(f"Subjects (baseline): {baseline_df['subject_id'].nunique()}")
    lines.append(f"Visits: {visit_df.shape[0]}")
    lines.append("")
    lines.append("## Label counts (baseline)")
    lines.append(baseline_df["label"].value_counts().to_string())
    lines.append("")
    lines.append("## Feature groups")
    for group, cols in feature_groups.items():
        lines.append(f"- {group}: {len(cols)} features")
    lines.append("")
    lines.append("## Missingness (baseline, mean per group)")
    for group, cols in feature_groups.items():
        if not cols:
            continue
        missing_rate = baseline_df[cols].isna().mean().mean()
        lines.append(f"- {group}: {missing_rate:.3f}")
    path.write_text("\n".join(lines))
