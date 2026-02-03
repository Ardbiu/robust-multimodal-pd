from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
import pandas as pd
from typing import Tuple, Dict, Generator
from pd_fusion.data.schema import TARGET_COL

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGK = True
except Exception:
    _HAS_SGK = False

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

def get_kfold_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Yields (train_df, val_df) for K-Fold CV.
    Note: For small datasets, we might use val_df as test_df (or nested CV).
    For this benchmark, we treat Val as the evaluation set for the fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df[TARGET_COL]
    
    for train_idx, val_idx in skf.split(df, y):
        yield df.iloc[train_idx], df.iloc[val_idx]

def get_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "subject_id",
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Yields (train_df, val_df) for GroupKFold CV.
    Uses StratifiedGroupKFold if available.
    """
    y = df[TARGET_COL]
    groups = df[group_col]

    if _HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in splitter.split(df, y, groups):
            yield df.iloc[train_idx], df.iloc[val_idx]

def split_train_calibration(
    df: pd.DataFrame,
    calib_size: float = 0.2,
    seed: int = 42,
    group_col: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a training dataframe into train/calibration sets.
    Uses group-aware split if group_col is provided.
    """
    if calib_size <= 0 or calib_size >= 1:
        raise ValueError("calib_size must be between 0 and 1.")
    if group_col and group_col in df.columns:
        y = df[TARGET_COL]
        groups = df[group_col]
        n_splits = max(2, int(round(1.0 / calib_size)))
        if _HAS_SGK:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            train_idx, calib_idx = next(splitter.split(df, y, groups))
        else:
            splitter = GroupKFold(n_splits=n_splits)
            train_idx, calib_idx = next(splitter.split(df, y, groups))
        return df.iloc[train_idx], df.iloc[calib_idx]
    # Fallback to stratified split
    train_df, calib_df = train_test_split(
        df, test_size=calib_size, stratify=df[TARGET_COL], random_state=seed
    )
    return train_df, calib_df
    else:
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in splitter.split(df, y, groups):
            yield df.iloc[train_idx], df.iloc[val_idx]

def get_subset_masks(maskdict: Dict, indices: pd.Index):
    # Maskdict arrays are aligned with original df indices if RangeIndex(0..N)
    # If df has RangeIndex, indices are integers.
    # If maskdict is simple arrays, we use integer indexing.
    # Check if indices match array shape.
    
    # Assumption: maskdict values are numpy arrays of length len(original_df)
    # And df.index corresponds to positions in these arrays (if RangeIndex)
    # OR we need array-indexing by position.
    
    # Safest: Use df.index to slice if mask was built aligned with df.
    # Current code assumes integer location indexing which works for RangeIndex.
    # If df was shuffled/split, indices might be permuted integers.
    # return {k: v[indices] for k, v in maskdict.items()} works if v is numpy array and indices is int array/list
    
    return {k: v[indices] for k, v in maskdict.items()}
