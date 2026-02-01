from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple, Dict
from pd_fusion.data.schema import TARGET_COL

def stratified_split(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    """
    Split data into train, val, test while preserving PD/HC ratio.
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=seed
    )
    
    # Adjust val_size to be relative to the original size
    rel_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=rel_val_size, stratify=train_val_df[TARGET_COL], random_state=seed
    )
    
    return train_df, val_df, test_df

def get_subset_masks(maskdict: Dict, indices: pd.Index):
    return {k: v[indices] for k, v in maskdict.items()}
