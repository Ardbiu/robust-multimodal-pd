import torch
import torch.nn as nn
import numpy as np
from pd_fusion.models.fusion_late import LateFusionModel, LateFusionNet

class MaskedFusionModel(LateFusionModel):
    """
    Appends the availability mask to the input features.
    """
    def __init__(self, input_dim, mask_dim, params):
        super().__init__(input_dim + mask_dim, params)
        self.mask_dim = mask_dim
        
    def train(self, X, y, val_data=None):
        # X and masks are concatenated in the training loop implementation
        super().train(X, y, val_data)
        
    def predict_proba(self, X, masks=None):
        if masks is not None:
            X = np.concatenate([X, masks], axis=1)
        return super().predict_proba(X)
