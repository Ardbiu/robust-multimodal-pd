import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pd_fusion.data.schema import MODALITIES, MODALITY_FEATURES, TARGET_COL, ID_COL

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
from pd_fusion.data.schema import MODALITIES, TARGET_COL, ID_COL
from pd_fusion.data.column_mapping import load_and_validate_raw_data
from pd_fusion.paths import PROCESSED_DATA_DIR

def load_ppmi_data(config: Dict, synthetic: bool = False) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Loads PPMI data from CSVs or generates synthetic data.
    Returns:
        df: Merged dataframe with all modalities.
        masks: Dictionary mapping modality to boolean availability mask.
    """
    if synthetic:
        return generate_synthetic_data(config["synthetic"])
    
    # Real data loading
    # Expects that 'validate-data' has been run or we run it on the fly?
    # Usually we load the processed parquet if available, else process.
    processed_path = PROCESSED_DATA_DIR / "ppmi_merged.parquet"
    if processed_path.exists():
        logging.getLogger("pd_fusion").info(f"Loading processed data from {processed_path}")
        df = pd.read_parquet(processed_path)
        # Reconstruct masks from columns (assuming we have a way to know)
        # Or better: we should save/load masks too, or re-derive them from NaNs.
        masks = create_masks_from_df(df, config["modalities"])
        return df, masks
    
    raise FileNotFoundError(f"Processed data not found at {processed_path}. Run 'validate-data' first.")

def process_and_merge_data(data_config: Dict, column_config: Dict):
    """
    Main entry point for 'validate-data' CLI.
    Loads raw CSVs, maps columns, merges, prints stats, and saves.
    """
    logger = logging.getLogger("pd_fusion")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load and Map
    raw_dfs = load_and_validate_raw_data(data_config, column_config)
    
    if not raw_dfs:
        logger.error("No valid data loaded from raw files.")
        return

    # 2. Merge (Outer Join to keep all subjects)
    # Start with Clinical as anchor or union of all PATNOs?
    # Strategy: Outer merge on patno (and possibly event_id).
    # For baseline prediction, we usually filter to BL event.
    
    merged_df = None
    
    for mod, df in raw_dfs.items():
        # Filter for baseline if relevant event_id column exists
        if "event_id" in df.columns:
            # Common strict baseline codes: 'BL', 'V01', 'SC' depending on study
            # For this skeleton, let's assume we keep all or filter later.
            # But robust fusion usually implies aligning by subject.
            # Let's assume input CSVs are already filtered or we group by PATNO.
            pass

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="patno", how="outer", suffixes=("", f"_{mod}"))

    # 3. Stats & Missingness
    logger.info(f"Merged DataFrame Shape: {merged_df.shape}")
    logger.info("Missingness Stats per Modality (based on key columns):")
    
    # Check presence based on mapped columns
    for mod in MODALITIES:
        # Determine "presence" by checking if key features are non-null
        # This requires knowing which columns belong to which modality
        # We can approximate by checking if ANY column from that modality (in raw_dfs) is present
        if mod in raw_dfs:
            # Which subjects came from this modality?
            # We can check intersection of indices if we set index to patno
            in_mod = raw_dfs[mod]["patno"].unique()
            n_present = len(in_mod)
            n_total = len(merged_df)
            logger.info(f"  {mod}: {n_present}/{n_total} ({n_present/n_total:.1%}) subjects present")

    # 4. Save
    out_path = PROCESSED_DATA_DIR / "ppmi_merged.parquet"
    merged_df.to_parquet(out_path)
    logger.info(f"Saved merged data to {out_path}")

def create_masks_from_df(df: pd.DataFrame, mod_config: Dict) -> Dict[str, np.ndarray]:
    """
    Derive masks based on NaNs in columns. 
    Assumes if key columns are missing, the modality is missing.
    """
    masks = {}
    # Need to know which columns belong to which modality.
    # In a real scenario, we'd prefix columns or track schema.
    # Here let's assume we can infer from prefixes or config.
    
    # Fallback: check schema? 
    # For now, let's look for known prefixes/mapped names from config if possible
    # Or just use the fact that we merged them. 
    # Actually, simpler: check if specific marker features are NaN.
    
    # But wait, we don't track column provenance easily in the merged DF unless we prefixed.
    # Let's rely on MODALITIES list and crude heuristic or prefix.
    # Skeleton heuristic: use schema definitions
    
    from pd_fusion.data.schema import MODALITY_FEATURES
    
    for mod in MODALITIES:
        # Check if ANY of the feature columns for this modality are present and non-NAN
        # But schema has placeholders. 
        # Better: use the column_map keys from config?
        # Let's rely on prefixes if we added them, or specific known columns.
        
        # Simple heuristic for this skeleton:
        # If 'clinical' -> check 'updrs_iii'
        # If 'datspect' -> check 'sbr_mean' or 'caudate_r'
        # If 'mri' -> check 'hippocampus_l' (if mapped)
        
        cols_to_check = []
        if mod == "clinical": cols_to_check = ["updrs_iii", "age"]
        elif mod == "datspect": cols_to_check = ["sbr_mean", "caudate_r"]
        elif mod == "mri": cols_to_check = ["hippocampus_l", "hippocampus_r"]
        
        present_mask = np.zeros(len(df), dtype=int)
        
        relevant_cols = [c for c in cols_to_check if c in df.columns]
        if relevant_cols:
            # Mark as present if at least one key column is not null?
            # Or stringent: all? Let's say at least one for "available"
            present_mask = df[relevant_cols].notna().any(axis=1).astype(int).values
            
        masks[mod] = present_mask
        
    return masks

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
