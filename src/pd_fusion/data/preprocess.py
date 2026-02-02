import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict

class NaNRobustScaler:
    """
    Robust Scaler that handles NaNs by computing stats ignoring them,
    and propagating them in transform.
    """
    def __init__(self):
        self.medians = None
        self.iqrs = None
        
    def fit(self, X: np.ndarray):
        # Calc stats ignoring NaNs
        self.medians = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.iqrs = q75 - q25
        # Avoid div by zero
        self.iqrs[self.iqrs == 0] = 1.0
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians is None:
            raise ValueError("Scaler not fitted")
        return (X - self.medians) / self.iqrs

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
    2. SCALES data using robust statistics (ignoring NaNs).
       CRITICAL: Does NOT impute missing values. Returns NaNs.
       Downstream dataset class uses NaNs to generate MASKS, then fills with 0.
    """
    # Filter to existing columns
    existing = [c for c in feature_cols if c in df.columns]
    
    # If missing all columns, return NaNs (representing missing modality)
    if not existing:
        return np.full((len(df), len(feature_cols)), np.nan), imputer, scaler
        
    X_df = df[feature_cols].copy()
    
    # Ensure all columns exist (fill missing cols with NaN if partial)
    # This handles case where 'feature_cols' has more than 'existing'
    for col in feature_cols:
        if col not in X_df:
            X_df[col] = np.nan
            
    X = X_df[feature_cols].values # Force order
    
    # Fit/Transform Scaler
    if scaler is None:
        scaler = NaNRobustScaler()
        scaler.fit(X)
        
    X_scaled = scaler.transform(X)
    
    # We return 'imputer' as None since we don't use it anymore, 
    # but keep signature for compatibility if needed.
    
    return X_scaled, None, scaler
