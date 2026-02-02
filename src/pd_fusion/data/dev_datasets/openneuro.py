import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from pd_fusion.data.schema import TARGET_COL, ID_COL
from pd_fusion.paths import DEV_DATA_DIR, ROOT_DIR
from pd_fusion.utils.io import load_yaml

logger = logging.getLogger("pd_fusion.openneuro")

DEFAULT_LABEL_CANDIDATES = [
    "group", "diagnosis", "dx", "phenotype", "status", "case_control", "patient"
]

DEFAULT_LABEL_MAP = {
    "pd": 1,
    "parkinson": 1,
    "parkinson's": 1,
    "patient": 1,
    "case": 1,
    "hc": 0,
    "control": 0,
    "healthy": 0,
    "ctl": 0,
}

def _load_label_config() -> Dict:
    cfg_path = ROOT_DIR / "configs" / "openneuro_labels.yaml"
    if cfg_path.exists():
        try:
            return load_yaml(cfg_path)
        except Exception as e:
            logger.warning(f"Failed to load openneuro label config: {e}")
    return {}

def _normalize_label(val, label_map: Dict[str, int]) -> Optional[int]:
    if pd.isna(val):
        return None
    # numeric labels
    if isinstance(val, (int, np.integer)):
        if int(val) in [0, 1]:
            return int(val)
    if isinstance(val, (float, np.floating)):
        if int(val) in [0, 1] and abs(val - int(val)) < 1e-6:
            return int(val)
    s = str(val).strip().lower()
    if s in label_map:
        return int(label_map[s])
    return None

def _infer_label_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _encode_sex(val) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ["m", "male", "1"]:
        return 1
    if s in ["f", "female", "0"]:
        return 0
    return None

def _build_clinical_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    clinical = pd.DataFrame({ID_COL: df[ID_COL].values})
    for col in df.columns:
        if col in [ID_COL, label_col]:
            continue
        if col.lower() in ["sex", "gender"]:
            clinical[f"clinical_{col.lower()}"] = df[col].apply(_encode_sex)
            continue
        # Try numeric conversion for general columns (age, scores, etc.)
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            clinical[f"clinical_{col}"] = series
    return clinical

def _count_mri_files(sub_dir: Path) -> Dict[str, int]:
    counts = {
        "t1w": 0,
        "t2w": 0,
        "bold": 0,
        "dwi": 0,
        "fmap": 0,
    }
    if not sub_dir.exists():
        return counts
    for path in sub_dir.rglob("*.nii*"):
        name = path.name.lower()
        if "_t1w" in name:
            counts["t1w"] += 1
        elif "_t2w" in name:
            counts["t2w"] += 1
        elif "_bold" in name:
            counts["bold"] += 1
        elif "_dwi" in name:
            counts["dwi"] += 1
    fmap_dir = sub_dir / "fmap"
    if fmap_dir.exists():
        counts["fmap"] = len(list(fmap_dir.rglob("*.nii*")))
    return counts

def _build_mri_proxy_features(root: Path, subject_ids: List[str]) -> pd.DataFrame:
    rows = []
    for sid in subject_ids:
        sub_id = sid if str(sid).startswith("sub-") else f"sub-{sid}"
        sub_dir = root / sub_id
        counts = _count_mri_files(sub_dir)
        rows.append({
            ID_COL: sid,
            "mri_t1w_count": counts["t1w"],
            "mri_t2w_count": counts["t2w"],
            "mri_bold_count": counts["bold"],
            "mri_dwi_count": counts["dwi"],
            "mri_fmap_count": counts["fmap"],
        })
    return pd.DataFrame(rows)

def load_openneuro_dataset(accession: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    dataset_root = DEV_DATA_DIR / "openneuro" / accession
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"OpenNeuro dataset not found at {dataset_root}. "
            "Run 'python -m pd_fusion.cli download-dev --dataset openneuro' or download manually."
        )
    participants_path = dataset_root / "participants.tsv"
    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")
    df = pd.read_csv(participants_path, sep="\t")

    # Subject ID normalization
    id_col = None
    for c in ["participant_id", "subject_id", "sub_id", "subject"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError("participants.tsv missing subject ID column (participant_id).")
    df = df.rename(columns={id_col: ID_COL})

    # Label config
    cfg = _load_label_config().get(accession, {})
    label_candidates = cfg.get("label_column_candidates", DEFAULT_LABEL_CANDIDATES)
    label_col = cfg.get("label_column") or _infer_label_column(df, label_candidates)
    if label_col is None:
        raise ValueError(
            f"Could not infer label column for {accession}. "
            "Update configs/openneuro_labels.yaml with label_column."
        )
    label_map = {k.lower(): v for k, v in cfg.get("label_map", DEFAULT_LABEL_MAP).items()}

    y = df[label_col].apply(lambda v: _normalize_label(v, label_map))
    keep = y.notna()
    df = df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(int).reset_index(drop=True)

    if y.nunique() < 2:
        raise ValueError(f"Label column '{label_col}' does not contain both classes for {accession}.")

    clinical_df = _build_clinical_features(df, label_col)
    mri_df = _build_mri_proxy_features(dataset_root, df[ID_COL].tolist())

    # Merge
    out = pd.DataFrame({ID_COL: df[ID_COL].values, TARGET_COL: y.values})
    out = out.merge(clinical_df, on=ID_COL, how="left")
    out = out.merge(mri_df, on=ID_COL, how="left")

    # Masks
    clinical_cols = [c for c in out.columns if c.startswith("clinical_")]
    mri_cols = [c for c in out.columns if c.startswith("mri_")]
    clinical_mask = (~out[clinical_cols].isna().all(axis=1)).astype(int).values if clinical_cols else np.zeros(len(out), dtype=int)
    mri_mask = (out[mri_cols].fillna(0).sum(axis=1) > 0).astype(int).values if mri_cols else np.zeros(len(out), dtype=int)
    if mri_cols and mri_mask.sum() == 0:
        logger.warning(f"No MRI files detected for {accession}; MRI modality will be absent.")

    # Set MRI features to NaN where modality absent
    if mri_cols:
        out.loc[mri_mask == 0, mri_cols] = np.nan

    masks = {
        "clinical": clinical_mask,
        "datspect": np.zeros(len(out), dtype=int),
        "mri": mri_mask,
    }
    return out, masks
