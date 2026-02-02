import numpy as np
import pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

class CalibratedModel:
    def __init__(self, base_model, method="isotonic"):
        self.base_model = base_model
        self.method = method
        self.calibrator = None
        
    def fit(self, X_val, y_val, masks_val=None):
        """
        Fits the calibrator on validation data.
        """
        # Get uncalibrated probabilities
        if "moe" in str(type(self.base_model)).lower():
            # Special handling for MoE inputs
            # Only support if X_val is properly formatted. 
            # If train.py handles calibration, it should pass compatible inputs.
            # Here we assume X_val is what predict_proba expects.
            preds = self.base_model.predict_proba(X_val, masks_val)
        elif hasattr(self.base_model, "predict_proba"):
             preds = self.base_model.predict_proba(X_val, masks_val)
        else:
            raise ValueError("Base model must have predict_proba")

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            self.calibrator = LogisticRegression() # Platt scaling
            
        self.calibrator.fit(preds, y_val)
        
    def predict_proba(self, X, masks=None):
        preds = self.base_model.predict_proba(X, masks)
        if self.calibrator:
            return self.calibrator.transform(preds)
        return preds
        
    def save(self, path):
        # We save both base model and calibrator
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
