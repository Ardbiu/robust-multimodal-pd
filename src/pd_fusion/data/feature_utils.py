from typing import List
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
