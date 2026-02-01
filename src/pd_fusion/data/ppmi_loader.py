import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pd_fusion.data.schema import MODALITIES, MODALITY_FEATURES, TARGET_COL, ID_COL

def load_ppmi_data(config: Dict, synthetic: bool = False) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Loads PPMI data from CSVs or generates synthetic data.
    Returns:
        df: Merged dataframe with all modalities.
        masks: Dictionary mapping modality to boolean availability mask.
    """
    if synthetic:
        return generate_synthetic_data(config["synthetic"])
    
    # Real data loading logic (skeleton)
    # df = merge_modalities(config["modalities"])
    # return df, create_masks(df)
    raise NotImplementedError("Real data loading requires CSVs in data/raw and implementation of merge_modalities.")

def generate_synthetic_data(synth_config: Dict) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    n = synth_config["num_samples"]
    data = {ID_COL: np.arange(n)}
    
    # Generate random features for each modality
    masks = {}
    for i, mod in enumerate(MODALITIES):
        dim = synth_config.get(f"{mod}_dim", 10)
        missing_rate = synth_config["missing_rates"][i]
        
        # Features
        feat_names = [f"{mod}_f{j}" for j in range(dim)]
        features = np.random.randn(n, dim)
        
        # Mask (1 = present, 0 = missing)
        mask = np.random.choice([0, 1], size=n, p=[missing_rate, 1 - missing_rate])
        masks[mod] = mask
        
        # Inject missingness (NaNs)
        features[mask == 0] = np.nan
        
        for j, name in enumerate(feat_names):
            data[name] = features[:, j]
            
    # Simple target generation dependent on clinical and datspect
    # PD = low datspect SBR, high clinical UPDRS
    clinical_score = data.get("clinical_f0", 0)
    dat_score = data.get("datspect_f0", 0)
    y_prob = 1 / (1 + np.exp(-(clinical_score - dat_score)))
    data[TARGET_COL] = (y_prob > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    return df, masks
