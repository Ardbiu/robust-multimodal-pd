import numpy as np
from pd_fusion.models.base import BaseModel
from pd_fusion.utils.io import save_pickle, load_pickle

class ConstantProbabilityModel(BaseModel):
    """
    Simple baseline that predicts a constant probability.
    Useful when a modality is entirely absent.
    """
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def train(self, X, y, val_data=None):
        self.p = float(np.mean(y)) if len(y) > 0 else 0.5

    def predict_proba(self, X, masks=None):
        n = len(X)
        return np.full(n, self.p)

    def save(self, path):
        save_pickle(self, path)

    @classmethod
    def load(cls, path):
        return load_pickle(path)
