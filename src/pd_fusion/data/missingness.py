import numpy as np
import pandas as pd
from typing import List, Dict

def apply_missingness_scenario(df: pd.DataFrame, scenario: Dict, maskdict: Dict[str, np.ndarray]):
    """
    Modifies the maskdict to simulate a specific missingness scenario.
    """
    new_masks = {k: v.copy() for k, v in maskdict.items()}
    
    if "drop_modalities" in scenario:
        for mod in scenario["drop_modalities"]:
            if mod in new_masks:
                new_masks[mod] = np.zeros_like(new_masks[mod])
                
    if scenario.get("type") == "random":
        # Randomly drop k modalities per subject
        n_drop = scenario.get("n_drop", 1)
        n_subjects = len(df)
        modalities = list(new_masks.keys())
        
        for i in range(n_subjects):
            choices = np.random.choice(modalities, size=n_drop, replace=False)
            for mod in choices:
                new_masks[mod][i] = 0
                
    return new_masks

def get_modality_mask_matrix(maskdict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Converts dict of masks to a binary matrix [N, M].
    """
    return np.stack([maskdict[m] for m in sorted(maskdict.keys())], axis=1)
