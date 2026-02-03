import pandas as pd
import numpy as np
from pd_fusion.utils.metrics import compute_metrics
from pd_fusion.data.schema import TARGET_COL, MODALITIES, MODALITY_FEATURES
from pd_fusion.data.missingness import apply_missingness_scenario
from pd_fusion.data.preprocess import preprocess_features
from pd_fusion.data.feature_utils import apply_masks_to_matrix
from pd_fusion.data.missingness import get_modality_mask_matrix
import torch

def evaluate_model(model, df_test, mask_test, prep_info, config):
    results = {}
    scenarios = config.get("scenarios", [{"name": "baseline", "drop_modalities": []}])
    
    y_true = df_test[TARGET_COL].values
    
    for scenario in scenarios:
        name = scenario["name"]
        # Apply missingness
        current_masks = apply_missingness_scenario(df_test, scenario, mask_test)
        
        # Prepare inputs
        inputs = None
        current_masks_tensor = None
        
        # Determine model type by checking prep_info structure
        # If prep_info is dict, it's MoE style (per-modality preprocessors)
        is_moe = isinstance(prep_info, dict)
        
        if is_moe:
            X_dict = {}
            mods_used = list(prep_info.keys())
            for mod in mods_used:
                imputer, scaler, feats = prep_info[mod]
                # Preprocess using fitted scaler
                X_mod, _, _ = preprocess_features(df_test, feats, imputer, scaler)
                if mod in current_masks:
                    X_mod = X_mod * current_masks[mod].reshape(-1, 1)
                X_dict[mod] = torch.FloatTensor(X_mod)
            inputs = X_dict
            current_masks_tensor = torch.FloatTensor(np.stack([current_masks[m] for m in mods_used], axis=1))
        else:
            # Standard fusion
            imputer, scaler, feature_cols = prep_info
            X_test, _, _ = preprocess_features(df_test, feature_cols, imputer, scaler)
            inputs = apply_masks_to_matrix(X_test, current_masks, feature_cols)
            
        # Predict
        if is_moe:
            y_prob = model.predict_proba(inputs, current_masks_tensor)
        else:
            # Pass masks if model accepts them (e.g. MaskedFusion, ModDrop)
            # MaskedFusion expects logic to append masks inside predict_proba if needed
            # But wait, predict_proba signature in base is (X, masks=None)
            # For Unimodal, masks might be ignored or used to zero out?
            # Ideally we pass masks dict.
            if hasattr(model, "mask_dim"):
                mask_mat = get_modality_mask_matrix(current_masks)
                y_prob = model.predict_proba(inputs, masks=mask_mat)
            else:
                y_prob = model.predict_proba(inputs, masks=current_masks)
            
        results[name] = compute_metrics(y_true, y_prob)
        
    return results

def compute_risk_coverage(y_true, y_prob, masks):
    """
    Computes Risk-Coverage curve by sweeping thresholds.
    This simulates what the Conformal Wrapper does but for analysis.
    For the Conformal Wrapper, we also have discrete abstain decisions.
    """
    # Simply, sort by confidence (max prob), then compute cum accuracy.
    # Confidence = max(p, 1-p)
    confidence = np.maximum(y_prob, 1-y_prob)
    
    # Sort descending
    indices = np.argsort(confidence)[::-1]
    y_sorted = y_true[indices]
    conf_sorted = confidence[indices]
    
    # Correct predictions
    # Predicted label: 1 if p>=0.5 else 0
    preds = (y_prob >= 0.5).astype(int)
    correct = (preds == y_true).astype(int)
    correct_sorted = correct[indices]
    
    # Cumulative stats
    n = len(y_true)
    coverage = np.arange(1, n + 1) / n
    # Cumulative correct
    cum_correct = np.cumsum(correct_sorted)
    accuracy = cum_correct / np.arange(1, n + 1)
    risk = 1 - accuracy
    
    return {"coverage": coverage, "risk": risk}
