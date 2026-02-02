import pandas as pd
from pd_fusion.data.schema import MODALITY_FEATURES

def get_mri_features(df: pd.DataFrame) -> pd.DataFrame:
    features = MODALITY_FEATURES["mri"]
    available_feats = [f for f in features if f in df.columns]
    
    # Optional: Normalize by ICV if present?
    # For now, just return tabular features
    return df[available_feats].copy()
