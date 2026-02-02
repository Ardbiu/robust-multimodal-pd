from typing import List, Dict

MODALITIES = ["clinical", "datspect", "mri"]

# Features updated to match canonical names in configs/ppmi_columns.yaml
MODALITY_FEATURES: Dict[str, List[str]] = {
    "clinical": ["age", "sex", "education", "updrs_iii", "disease_duration"],
    "datspect": ["caudate_l", "caudate_r", "putamen_l", "putamen_r", "sbr_mean"],
    "mri": ["hippocampus_l", "hippocampus_r"] # Add others as defined in mapping
}

TARGET_COL = "diagnosis" # 1 for PD, 0 for HC
ID_COL = "patno"
