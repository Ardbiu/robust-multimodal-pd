import pandas as pd
import logging
from typing import Dict, List, Optional
from pathlib import Path

class ColumnMapper:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("pd_fusion")

    def validate_and_map(self, df: pd.DataFrame, modality: str) -> Optional[pd.DataFrame]:
        """
        Validates presence of required columns and maps them to canonical names.
        Returns mapped DataFrame or None if validation fails.
        """
        if modality not in self.config:
            self.logger.warning(f"No configuration found for modality: {modality}")
            return None

        mod_config = self.config[modality]
        required_cols = mod_config.get("required_columns", [])
        column_map = mod_config.get("column_map", {})

        # Check for missing columns (considering the map keys might be what's in the DF)
        # We need to see if the required raw columns (or their mapped versions) are present?
        # Actually usually 'required_columns' in config refers to the RAW names we expect.
        
        missing_cols = [col for col in required_cols if col not in df.columns and col not in column_map.values()]
        
        # If strict validation of raw names:
        # missing_cols = [col for col in required_cols if col not in df.columns]
        
        # However, we might have renamed them already or user provided canonical names.
        # Let's assume we are checking RAW headers.
        
        # Refined logic: check if 'required_columns' are in df key space.
        # But wait, config required_columns usually implies the INPUT requirement.
        
        final_missing = []
        for req in required_cols:
            if req in df.columns:
                continue
            # Maybe it's mapped?
            # if we have a map RAW->CANONICAL, we are looking for RAW in df.
            final_missing.append(req)

        if final_missing:
            self.logger.error(f"Missing required columns for {modality}: {final_missing}")
            # we might want to fail hard or return partial
            # For this strict pipeline, let's return None or raise
            return None

        # Apply mapping
        # Only map columns that are present and defined in the map
        rename_dict = {k: v for k, v in column_map.items() if k in df.columns}
        df_mapped = df.rename(columns=rename_dict)
        
        # Keep canonical columns (mapped ones + any other relevant ones if we want to filter)
        # For now, let's just return the renamed dataframe, maybe filtering to just relevant ones later
        
        return df_mapped

def load_and_validate_raw_data(data_config: Dict, column_config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Loads raw CSVs defined in data_config, validates/maps them using column_config.
    """
    logger = logging.getLogger("pd_fusion")
    raw_dir = Path(data_config["raw_data_dir"])
    mapper = ColumnMapper(column_config)
    
    loaded_data = {}
    
    for mod in data_config["modalities"]:
        files = data_config["modalities"][mod]["files"]
        # Allow multiple files per modality (e.g. merge them? or just pick first?)
        # For this skeleton, assume we might concat or merge if multiple. 
        # But let's start simple: assume 1 main file or handle list.
        
        dfs = []
        for f_name in files:
            f_path = raw_dir / f_name
            if not f_path.exists():
                logger.error(f"File not found: {f_path}")
                continue
            
            try:
                df = pd.read_csv(f_path)
                mapped = mapper.validate_and_map(df, mod)
                if mapped is not None:
                    dfs.append(mapped)
            except Exception as e:
                logger.error(f"Error loading {f_path}: {e}")

        if dfs:
            # Merge if multiple (on PATNO/EVENT_ID? logic specific to modality)
            # For now, take the first one or concat
            loaded_data[mod] = dfs[0] 
            if len(dfs) > 1:
                logger.warning(f"Multiple files loaded for {mod}, using first one only for now.")
        else:
            logger.warning(f"No valid data loaded for modality: {mod}")
            
    return loaded_data
