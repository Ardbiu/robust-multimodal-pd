import pandas as pd
import numpy as np
from pd_fusion.utils.metrics import compute_metrics
from pd_fusion.data.schema import TARGET_COL, MODALITIES
from pd_fusion.data.missingness import apply_missingness_scenario
from pd_fusion.data.preprocess import preprocess_features
import torch

def evaluate_model(model, df_test, mask_test, preprocess_info, config):
    results = {}
    scenarios = config.get("scenarios", [{"name": "baseline", "drop_modalities": []}])
    
    imputer, scaler, feature_cols = preprocess_info
    y_true = df_test[TARGET_COL].values
    
    for scenario in scenarios:
        name = scenario["name"]
        # Apply missingness
        current_masks = apply_missingness_scenario(df_test, scenario, mask_test)
        
        # Prepare inputs based on model expectations
        # This is a bit complex as different models expect different inputs
        # Here we provide a simplified version
        
        if hasattr(model, "mod_name"): # Unimodal
            X_test, _, _ = preprocess_features(df_test, feature_cols, imputer, scaler)
            # Apply mask conceptually? GBDTs handle NaNs, but we can zero out if explicitly requested
            y_prob = model.predict_proba(X_test)
        elif "moe" in str(type(model)).lower():
            X_dict = {}
            for mod in MODALITIES:
                feats = [col for col in df_test.columns if col.startswith(f"{mod}_")]
                X_mod, _, _ = preprocess_features(df_test, feats)
                X_dict[mod] = torch.FloatTensor(X_mod)
            m_tensor = torch.FloatTensor(np.stack([current_masks[m] for m in MODALITIES], axis=1))
            y_prob = model.predict_proba(X_dict, m_tensor)
        else: # Fusion
            X_test, _, _ = preprocess_features(df_test, feature_cols, imputer, scaler)
            y_prob = model.predict_proba(X_test)
            
        results[name] = compute_metrics(y_true, y_prob)
        
    return results
活跃的
