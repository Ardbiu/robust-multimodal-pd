import pandas as pd
import numpy as np
from typing import List
# Note: MODALITY_FEATURES is likely defined in schema.py based on ppmi_columns.yaml
# But we can also rely on column_mapping having already renamed them.

def get_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts clinical features defined in schema.
    Handles categorical encoding (e.g. Sex) and type coercion.
    """
    # Define features to extract (canonical names after mapping)
    features = ["updrs_iii", "age", "sex", "education", "duration_yr"]
    
    # Filter to what is actually present
    available_feats = [f for f in features if f in df.columns]
    
    subset = df[available_feats].copy()
    
    # Preprocessing specific to clinical data
    if "sex" in subset.columns:
        # Standardize Sex: 0=F, 1=M. 
        # PPMI raw often has F=0, M=1? Or strings.
        # Robust conversion:
        subset["sex"] = subset["sex"].apply(lambda x: 1 if str(x).upper() in ['M', '1', '1.0'] else 0 if str(x).upper() in ['F', '0', '0.0'] else np.nan)
        
    if "updrs_iii" in subset.columns:
        subset["updrs_iii"] = pd.to_numeric(subset["updrs_iii"], errors="coerce")
        
    if "age" in subset.columns:
        subset["age"] = pd.to_numeric(subset["age"], errors="coerce")
        
    return subset
