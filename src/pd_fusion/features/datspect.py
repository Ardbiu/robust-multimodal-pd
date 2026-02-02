import pandas as pd
import numpy as np

def get_datspect_features(df: pd.DataFrame) -> pd.DataFrame:
    # ROI columns expected after mapping
    roi_cols = ["caudate_r", "caudate_l", "putamen_r", "putamen_l", "sbr_mean"]
    available_feats = [f for f in roi_cols if f in df.columns]
    
    subset = df[available_feats].copy()
    
    # Calculate asymmetry: |L-R| / mean(L,R)
    # 0 = symmetric, higher = more asymmetric (typical in PD)
    
    if "caudate_l" in subset.columns and "caudate_r" in subset.columns:
        mean_caudate = (subset["caudate_l"] + subset["caudate_r"]) / 2.0
        subset["caudate_asym"] = (subset["caudate_l"] - subset["caudate_r"]).abs() / (mean_caudate + 1e-6)
        
    if "putamen_l" in subset.columns and "putamen_r" in subset.columns:
        mean_putamen = (subset["putamen_l"] + subset["putamen_r"]) / 2.0
        subset["putamen_asym"] = (subset["putamen_l"] - subset["putamen_r"]).abs() / (mean_putamen + 1e-6)
        
    return subset
