import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import List

def preprocess_features(df: pd.DataFrame, feature_cols: List[str], imputer=None, scaler=None):
    """
    Impute and scale features. 
    Note: For multimodal fusion, we usually keep NaNs to be handled by masks or specific models.
    """
    X = df[feature_cols].copy()
    
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
    else:
        X_imputed = imputer.transform(X)
        
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = scaler.transform(X_imputed)
        
    return X_scaled, imputer, scaler
