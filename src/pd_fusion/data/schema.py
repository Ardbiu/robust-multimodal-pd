from typing import List, Dict

MODALITIES = ["clinical", "datspect", "mri"]

MODALITY_FEATURES: Dict[str, List[str]] = {
    "clinical": ["age", "gender", "updrs_iii_score", "disease_duration", "education_years"], # Placeholder
    "datspect": ["caudate_l", "caudate_r", "putamen_l", "putamen_r", "sbr_ratio"], # Placeholder
    "mri": ["hippocampus_vol", "entorhinal_thick", "ventricle_vol", "gray_matter_vol"] # Placeholder
}

TARGET_COL = "diagnosis" # 1 for PD, 0 for HC
ID_COL = "patno"
