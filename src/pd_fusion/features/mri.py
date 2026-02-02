import pandas as pd
import numpy as np

def get_mri_features(df: pd.DataFrame) -> pd.DataFrame:
    # Common FreeSurfer ROIs for PD
    # We expect these to be mapped from raw columns
    # E.g. "hippocampus_l", "hippocampus_r", "icv"
    
    # For this version, let's grab all columns that match our mapped set?
    # Or just grab everything available that isn't metadata?
    # Let's be specific if we can, or permissive.
    
    # Assuming the df passed here already has CANONICAL names from mapper.
    cols_to_keep = [c for c in df.columns if c not in ["patno", "event_id", "date"]]
    subset = df[cols_to_keep].copy()
    
    # Normalize by Intracranial Volume (ICV) if available
    if "icv" in subset.columns:
        icv = subset["icv"]
        # Don't normalize ICV itself or non-volumetric flags
        for col in subset.columns:
            if col != "icv" and pd.api.types.is_numeric_dtype(subset[col]):
                # Simple ratio normalization
                subset[col] = subset[col] / (icv + 1e-6)
                
    return subset
