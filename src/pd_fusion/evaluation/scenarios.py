from typing import List, Dict
import numpy as np

def get_scenarios() -> List[Dict]:
    """
    Returns a list of scenarios to evaluate.
    """
    scenarios = [
        {"name": "full_observation", "drop_modalities": []},
        {"name": "missing_dat", "drop_modalities": ["datspect"]},
        {"name": "missing_mri", "drop_modalities": ["mri"]},
        {"name": "clinical_only", "drop_modalities": ["datspect", "mri"]},
        {"name": "random_1_drop_stress", "type": "random", "n_drop": 1},
        {"name": "random_2_drop_stress", "type": "random", "n_drop": 2},
    ]
    return scenarios

def get_custom_scenario(name):
    all_scens = {s["name"]: s for s in get_scenarios()}
    return all_scens.get(name, {"name": name, "drop_modalities": []})
