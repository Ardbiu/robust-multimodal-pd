import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from typing import List, Tuple, Optional

def preprocess_features(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    imputer=None, 
    scaler=None,
    strategy: str = "robust" 
) -> Tuple[np.ndarray, object, object]:
    """
    Preprocesses features:
    1. Selects columns.
    2. Imputes MISSING values WITHIN the modality (e.g. mean/median).
       CRITICAL: Does NOT impute completely missing modalities (those are handled by masks).
       However, if we pass a DF that has NaNs for missing modalities, 
       standard imputation effectively imputes them. 
       
       For Multimodal Fusion with Masks:
       We usually want to keep NaNs for missing modalities so the model can see them 
       or we zero-fill them and let the mask tell the story.
       
       Here, we employ a "Zero-fill + Scale" strategy for missing modalities,
       assuming the Mask will carry the signal. 
       OR we use mean-imputation for everything.
       
       Research decision: 
       - SimpleImputer(median) will fill NaNs. 
       - If a subject is missing M1, all M1 feats are NaN. Imputer fills with median of existing M1.
       - This is "mean imputation" for missing modalities.
    """
    # Filter to existing columns
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        return np.zeros((len(df), len(feature_cols))), imputer, scaler
        
    X = df[feature_cols].copy()
    
    # 1. Impute
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
    else:
        # Imputer expects same cols. If test set is missing some, we have issues.
        # Ideally we ensure columns align.
        X_imputed = imputer.transform(X)
        
    # 2. Scale
    if scaler is None:
        if strategy == "robust":
            scaler = RobustScaler()
        elif strategy == "quantile":
            scaler = QuantileTransformer(output_distribution="normal")
        else:
            scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = scaler.transform(X_imputed)
        
    return X_scaled, imputer, scaler
