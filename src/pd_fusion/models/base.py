from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y, val_data=None):
        pass
    
    @abstractmethod
    def predict_proba(self, X, masks=None):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass
