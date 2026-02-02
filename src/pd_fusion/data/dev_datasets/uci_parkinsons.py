import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from pd_fusion.data.schema import MODALITIES, TARGET_COL, ID_COL

def load_uci_parkinsons() -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Loads UCI Parkinsons dataset and adapts it to the canonical 
    multimodal format (X_clin, X_dat, X_mri, mask, y).
    
    Source: data/raw_dev/uci/parkinsons.data
    """
    from pd_fusion.paths import DEV_DATA_DIR
    data_path = DEV_DATA_DIR / "uci" / "parkinsons.data"
    
    if not data_path.exists():
        raise FileNotFoundError(f"UCI Parkinsons data not found at {data_path}. Run 'python -m pd_fusion.cli download-dev' first.")
        
    df = pd.read_csv(data_path)
    
    # UCI Parkinsons columns:
    # name, MDVP:Fo(Hz), ..., status, ...
    
    # 1. Standardize Target and ID
    df = df.rename(columns={"status": TARGET_COL, "name": ID_COL})
    
    # 2. Prefix features as 'clinical_'
    # We treat all voice features as clinical for this pipeline
    modality = "clinical"
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]
    
    rename_map = {c: f"{modality}_{c}" for c in feature_cols}
    df = df.rename(columns=rename_map)
    
    # 3. Create Masks
    # Clinical is available for everyone (mask=1)
    # DatSpect/MRI are missing for everyone (mask=0)
    masks = {}
    n = len(df)
    
    masks["clinical"] = np.ones(n, dtype=int)
    masks["datspect"] = np.zeros(n, dtype=int)
    masks["mri"] = np.zeros(n, dtype=int)
    
    return df, masks
