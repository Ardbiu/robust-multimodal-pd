import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from pd_fusion.data.schema import MODALITIES, TARGET_COL, ID_COL

def load_uci_telemonitoring() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Loads UCI Telemonitoring dataset.
    NOTE: This dataset contains only PD patients (regression task for UPDRS).
    It is not suitable for PD vs HC classification without external controls.
    
    Source: data/raw_dev/uci/parkinsons_updrs.data
    """
    from pd_fusion.paths import DEV_DATA_DIR
    data_path = DEV_DATA_DIR / "uci" / "parkinsons_updrs.data"
    
    if not data_path.exists():
        raise FileNotFoundError(f"UCI Telemonitoring data not found at {data_path}. Run 'python -m pd_fusion.cli download-dev' first.")
        
    df = pd.read_csv(data_path)
    
    # Columns: subject#, age, sex, test_time, motor_UPDRS, total_UPDRS, Jitter..., Shimmer...
    
    # 1. Standardize ID
    df = df.rename(columns={"subject#": ID_COL})
    
    # 2. Target
    # This dataset is PD-only; create a binary severity proxy for classification.
    # Use total_UPDRS if available; otherwise motor_UPDRS.
    severity_col = "total_UPDRS" if "total_UPDRS" in df.columns else "motor_UPDRS"
    if severity_col not in df.columns:
        raise ValueError("Telemonitoring dataset missing UPDRS columns for severity proxy.")
    median_val = df[severity_col].median()
    df[TARGET_COL] = (df[severity_col] >= median_val).astype(int)
    
    # 3. Prefix features
    modality = "clinical"
    # Treat voice + age/sex as clinical
    exclude = [ID_COL, TARGET_COL, "motor_UPDRS", "total_UPDRS"]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    rename_map = {c: f"{modality}_{c}" for c in feature_cols}
    df = df.rename(columns=rename_map)
    
    # 4. Masks
    masks = {}
    n = len(df)
    masks["clinical"] = np.ones(n, dtype=int)
    masks["datspect"] = np.zeros(n, dtype=int)
    masks["mri"] = np.zeros(n, dtype=int)
    
    return df, masks
