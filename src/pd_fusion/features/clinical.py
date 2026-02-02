import pandas as pd
from typing import List
from pd_fusion.data.schema import MODALITY_FEATURES

def get_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts clinical features defined in schema.
    Handles categorical encoding (e.g. Sex) and type coercion.
    """
    features = MODALITY_FEATURES["clinical"]
    # Check if we have these columns in the dataframe
    available_feats = [f for f in features if f in df.columns]
    
    subset = df[available_feats].copy()
    
    # Preprocessing specific to clinical data
    if "sex" in subset.columns:
        # Encode Sex: F -> 0, M -> 1 (or similar)
        # Assuming M/F string or 1/2 numeric. PPMI often uses 0=F, 1=M or 1/2.
        # Let's standardize to numeric
        subset["sex"] = pd.to_numeric(subset["sex"], errors="coerce")
        
    return subset
