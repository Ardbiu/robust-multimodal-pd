from typing import List, Dict
import numpy as np
import pandas as pd
from pd_fusion.data.schema import MODALITIES, MODALITY_FEATURES

def get_modality_feature_cols(df: pd.DataFrame, modality: str) -> List[str]:
    """
    Resolve feature columns for a modality.
    - If prefixed columns exist (e.g., clinical_age), use them.
    - Otherwise fall back to canonical schema columns (e.g., age, sex).
    """
    prefixed = [c for c in df.columns if c.startswith(f"{modality}_")]
    if prefixed:
        return prefixed
    return [c for c in MODALITY_FEATURES.get(modality, []) if c in df.columns]

def get_all_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Concatenate modality feature columns in fixed MODALITIES order.
    """
    all_features: List[str] = []
    for mod in MODALITIES:
        all_features.extend(get_modality_feature_cols(df, mod))
    return all_features

def get_feature_slices(feature_cols: List[str]) -> Dict[str, List[int]]:
    """
    Map feature column indices to modalities.
    Uses prefix matching (e.g., clinical_) first, then falls back to schema names.
    """
    slices: Dict[str, List[int]] = {m: [] for m in MODALITIES}
    for i, col in enumerate(feature_cols):
        assigned = False
        for mod in MODALITIES:
            if col.startswith(f"{mod}_"):
                slices[mod].append(i)
                assigned = True
                break
        if assigned:
            continue
        for mod, feats in MODALITY_FEATURES.items():
            if col in feats:
                slices[mod].append(i)
                assigned = True
                break
        # Unknown columns are left unassigned
    return slices

def apply_masks_to_matrix(X: np.ndarray, masks: Dict[str, np.ndarray], feature_cols: List[str]) -> np.ndarray:
    """
    Zero out feature blocks for modalities that are masked (0).
    """
    X_masked = X.copy()
    slices = get_feature_slices(feature_cols)
    for mod, idxs in slices.items():
        if not idxs:
            continue
        if mod in masks:
            mvec = masks[mod].reshape(-1, 1)
            X_masked[:, idxs] = X_masked[:, idxs] * mvec
    return X_masked
