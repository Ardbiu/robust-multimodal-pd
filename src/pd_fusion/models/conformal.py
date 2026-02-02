import numpy as np
import pickle
from typing import Dict, Tuple, List, Union

class MaskConformalWrapper:
    """
    Mask-Conditioned Conformal Prediction Wrapper.
    Learns a separate rejection threshold for each unique modality mask pattern.
    """
    def __init__(self, base_model, alpha: float = 0.1):
        self.base_model = base_model
        self.alpha = alpha
        self.thresholds = {} # Map mask_key -> quantile
        self.global_threshold = 0.0 # Fallback
        
    def _mask_to_key(self, mask_row) -> str:
        """Converts boolean/int mask array to string key."""
        return "".join(map(str, mask_row.astype(int)))
        
    def fit(self, X_cal: Union[np.ndarray, Dict], y_cal: np.ndarray, masks_cal: Dict[str, np.ndarray]):
        """
        Calibrate thresholds on held-out calibration set.
        params:
            X_cal: Calibration features (array or dict for MoE)
            y_cal: True labels
            masks_cal: Dictionary of {modality: boolean_array}
        """
        # 1. Get predictions (probabilities)
        # Handle different input types (Dict for MoE, Array for Fusion)
        if isinstance(X_cal, dict): 
            # Need to format masks as tensor for MoE predict if required, 
            # but predict_proba usually handles it or we use wrapper logic.
            # Here we assume base_model.predict_proba handles the X_cal struct.
            # But we need to pass tensor masks if MoE. 
            # Let's rely on how we called it in evaluate.py
            # Actually, `base_model` might be the raw model or CalibratedModel.
            # If it's CalibratedModel, it handles standardizing inputs? 
            # No, CalibratedModel forwards args.
            
             # Reconstruct mask tensor for MoE if needed
             # For simplicity, try passing masks_cal directly if model handles it, 
             # or handle conversion here.
             # Base models in this repo (ModDrop, LateFusion) take (X, masks=None).
             # MoE takes (X_dict, mask_tensor).
             pass
             
        # Unified prediction call
        # We need a unified way to get probs. Let's try calling with keyword args if possible.
        # Use helper logic similar to evaluate.py
        
        # Simplified: We assume X_cal is already preprocessed appropriately OR we handle raw?
        # In run_experiment, we pass preprocessed data.
        
        # Helper to convert masks dict to tensor/array for model input
        # Note: We need the masks for grouping anyway.
        
        # Let's assume input is compatible with base_model.predict_proba
        try:
           probs = self.base_model.predict_proba(X_cal, masks=masks_cal) 
        except TypeError:
            # Maybe it is MoE expecting positional arg for masks
            # Or explicit mask tensor.
            # Let's assume we can handle this or failure will guide us.
            # For now support the standard interface we built.
            probs = self.base_model.predict_proba(X_cal)

        # 2. Compute non-conformity scores
        # s(x) = 1 - p_true_class(x)? 
        # Standard for binary classification:
        # If y=1, s = 1 - p. If y=0, s = 1 - (1-p) = p.
        # Basically s = 1 - prob_of_correct_class
        
        # Flatten probs
        probs = probs.flatten()
        
        scores = np.zeros_like(probs)
        scores[y_cal == 1] = 1 - probs[y_cal == 1]
        scores[y_cal == 0] = probs[y_cal == 0] # 1 - (1 - p) = p
        
        # 3. Group by mask
        # Need to align masks to rows.
        # masks_cal is {mod: [N]}. Stack them.
        mod_keys = sorted(masks_cal.keys())
        mask_matrix = np.stack([masks_cal[k] for k in mod_keys], axis=1) # [N, K]
        
        mask_groups = {}
        for i, row in enumerate(mask_matrix):
            key = self._mask_to_key(row)
            if key not in mask_groups:
                mask_groups[key] = []
            mask_groups[key].append(scores[i])
            
        # 4. Compute thresholds
        # Q = Quantile_{1-alpha}(scores)
        # If s <= Q, we accept.
        # Ideally using (N+1)/N correction for rigorous CP, but standard quantile is fine for now.
        
        for key, group_scores in mask_groups.items():
            n = len(group_scores)
            if n < 10: 
                # Fallback if too few samples?
                # Use global quantile or conservative 1.0 (always accept? no always abstain?)
                # 1.0 would accept everything (since s <= 1). 
                # We want q such that P(s <= q) >= 1-alpha.
                # If we have no data, maybe use global.
                continue
                
            q = np.quantile(group_scores, np.ceil((n+1)*(1-self.alpha))/n, method='higher', interpolation='higher') if hasattr(np.quantile, 'method') else np.quantile(group_scores, 1-self.alpha)
            # Simple percentile
            q = np.percentile(group_scores, (1-self.alpha)*100)
            self.thresholds[key] = q
            
        # Global fallback
        self.global_threshold = np.percentile(scores, (1-self.alpha)*100)
        
    def predict(self, X, masks) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predictions, abstention_mask).
        abstention_mask: True if abstained.
        predictions: The probability (or 0.5/NaN if abstained? keeping prob is useful)
        """
        # Get probs
        try:
            probs = self.base_model.predict_proba(X, masks=masks)
        except:
             # Try MoE signature manually if needed, or assume standard
             # Re-implementing simplified call
             probs = self.base_model.predict_proba(X)
        
        probs = probs.flatten()
        pred_labels = (probs >= 0.5).astype(int)
        
        # Compute scores for the *predicted* label
        # At test time: s = 1 - p_predicted
        # if p >= 0.5, pred=1, s = 1-p.
        # if p < 0.5, pred=0, s = 1-(1-p) = p.
        # Basically s = 1 - max(p, 1-p) ? 
        # Standard: s(x, y) needs candidate label. 
        # For "Selective Prediction" (risk control), we usually use:
        # Confidence = max(p, 1-p). Abstain if Confidence < Threshold.
        # This maps to conformal if we frame it as:
        # Nonconformity = 1 - max(p, 1-p).
        # We calibrated Q for "1 - p_true". 
        # At test time we check validity for both classes? 
        # Or just "Selective Classification"?
        # User asked for: "If model is confident enough... predict. Else abstain."
        # The prompt implies Selective Classification approach calibrated via CP.
        
        # Let's use the selective classification formulation:
        # Score = 1 - max(p, 1-p)  (Risk/Uncertainty)
        # Threshold Q from calibration such that P(Score <= Q) = 1 - alpha ? 
        # Not exactly.
        
        # Let's stick to the prompt's definition:
        # s(x) = 1 - p_y_hat(x)
        # q_m = Quantile(s_i on calib)
        # Test: accept if s(x) <= q_m
        
        # This guarantees that the *coverage* is controlled? NO.
        # It guarantees that *IF* we accepted, error is controlled? 
        # Actually standard CP guarantees coverage of the true label sets.
        # Selective prediction controls risk given coverage.
        
        # Let's implement exactly what is requested:
        # 1. Calc s(x) = 1 - p_hat (where p_hat is prob of predicted class)
        # 2. Check s(x) <= q_m
        
        scores = np.zeros_like(probs)
        # Predicted class 1 -> prob is p. s = 1-p.
        # Predicted class 0 -> prob is 1-p. s = 1-(1-p) = p.
        # So s = 1 - max(p, 1-p) = min(p, 1-p).
        scores = np.minimum(probs, 1-probs)
        
        # Check thresholds
        mod_keys = sorted(masks.keys())
        mask_matrix = np.stack([masks[k] for k in mod_keys], axis=1)
        
        abstain = np.zeros(len(probs), dtype=bool)
        
        for i, row in enumerate(mask_matrix):
            key = self._mask_to_key(row)
            thresh = self.thresholds.get(key, self.global_threshold)
            
            if scores[i] > thresh:
                abstain[i] = True
                
        return probs, abstain
        
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
