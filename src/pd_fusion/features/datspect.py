import pandas as pd
import numpy as np
from pd_fusion.data.schema import MODALITY_FEATURES

def get_datspect_features(df: pd.DataFrame) -> pd.DataFrame:
    features = MODALITY_FEATURES["datspect"]
    available_feats = [f for f in features if f in df.columns]
    subset = df[available_feats].copy()
    
    # Calculate asymmetry if raw left/right are present
    if "caudate_l" in subset.columns and "caudate_r" in subset.columns:
        subset["caudate_asym"] = (subset["caudate_r"] - subset["caudate_l"]) / (subset["caudate_r"] + subset["caudate_l"] + 1e-6)
        
    if "putamen_l" in subset.columns and "putamen_r" in subset.columns:
        subset["putamen_asym"] = (subset["putamen_r"] - subset["putamen_l"]) / (subset["putamen_r"] + subset["putamen_l"] + 1e-6)
        
    return subset
