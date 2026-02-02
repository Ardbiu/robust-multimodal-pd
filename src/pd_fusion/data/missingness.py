import numpy as np
import pandas as pd
import logging
from typing import List, Dict
from pd_fusion.data.schema import MODALITIES

def apply_missingness_scenario(df: pd.DataFrame, scenario: Dict, maskdict: Dict[str, np.ndarray]):
    """
    Modifies the maskdict to simulate a specific missingness scenario.
    """
    logger = logging.getLogger("pd_fusion")
    new_masks = {k: v.copy() for k, v in maskdict.items()}
    
    if "drop_modalities" in scenario:
        for mod in scenario["drop_modalities"]:
            if mod in new_masks:
                if np.all(new_masks[mod] == 0):
                    scen_name = scenario.get("name", "unnamed")
                    logger.info(f"[missingness] scenario '{scen_name}': modality '{mod}' already absent; no-op.")
                new_masks[mod] = np.zeros_like(new_masks[mod])
            else:
                scen_name = scenario.get("name", "unnamed")
                logger.info(f"[missingness] scenario '{scen_name}': modality '{mod}' not found in masks; no-op.")
                
    if scenario.get("type") == "random":
        # Randomly drop k modalities per subject
        n_drop = scenario.get("n_drop", 1)
        n_subjects = len(df)
        modalities = list(new_masks.keys()) if new_masks else MODALITIES
        
        for i in range(n_subjects):
            available = [m for m in modalities if m in new_masks and new_masks[m][i] == 1]
            if not available:
                continue
            drop_n = min(n_drop, len(available))
            choices = np.random.choice(available, size=drop_n, replace=False)
            for mod in choices:
                new_masks[mod][i] = 0
                
    return new_masks

def get_modality_mask_matrix(maskdict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Converts dict of masks to a binary matrix [N, M].
    """
    if not maskdict:
        raise ValueError("maskdict is empty")
    matrices = []
    # Preserve MODALITIES order for consistency
    for m in MODALITIES:
        if m in maskdict:
            matrices.append(maskdict[m])
        else:
            matrices.append(np.zeros_like(next(iter(maskdict.values()))))
    return np.stack(matrices, axis=1)
