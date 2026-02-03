import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pd_fusion.data.schema import TARGET_COL
from pd_fusion.data.openneuro_features import (
    load_simple_features,
    load_cnn_embeddings,
    load_resnet2d_embeddings,
)

def _resolve_manifest_path(config: Dict) -> Path:
    env_path = os.environ.get("PD_FUSION_DS001907_MANIFEST")
    if env_path:
        return Path(env_path)
    manifest = config.get("manifest_path", "data/processed/openneuro_ds001907_manifest.csv")
    return Path(manifest)

def load_openneuro_ds001907(config: Dict) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Load ds001907 using a prebuilt manifest. Returns dataframe with mri_ features.
    """
    manifest_path = _resolve_manifest_path(config)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    feature_mode = config.get("feature_mode", "simple")
    feature_cache_dir = config.get("feature_cache_dir", "data/processed/openneuro_ds001907/features_simple")
    embedding_cache_dir = config.get("embedding_cache_dir", "data/processed/openneuro_ds001907/embeddings_cnn3d")

    if feature_mode == "simple":
        df = load_simple_features(manifest_path, Path(feature_cache_dir), config.get("feature_config", {}))
    elif feature_mode == "cnn3d":
        df = load_cnn_embeddings(manifest_path, Path(embedding_cache_dir), config.get("cnn_config", {}))
    elif feature_mode == "resnet2d":
        resnet_cache_dir = config.get(
            "resnet2d_cache_dir",
            "data/processed/openneuro_ds001907/embeddings_resnet2d"
        )
        df = load_resnet2d_embeddings(
            manifest_path, Path(resnet_cache_dir), config.get("resnet2d_config", {})
        )
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    # Ensure label column name is canonical
    if "label" in df.columns and TARGET_COL not in df.columns:
        df[TARGET_COL] = df["label"].astype(int)

    # Masks: mri present if any mri_ feature is non-null
    mri_cols = [c for c in df.columns if c.startswith("mri_")]
    if not mri_cols:
        raise ValueError("No mri_ feature columns found in ds001907 dataframe.")
    mri_mask = (~df[mri_cols].isna().all(axis=1)).astype(int).values

    masks = {
        "clinical": np.zeros(len(df), dtype=int),
        "datspect": np.zeros(len(df), dtype=int),
        "mri": mri_mask,
    }
    return df, masks
