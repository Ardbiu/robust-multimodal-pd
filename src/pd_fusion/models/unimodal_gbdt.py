import lightgbm as lgb
import xgboost as xgb
from pd_fusion.models.base import BaseModel
from pd_fusion.utils.io import save_pickle, load_pickle

class UnimodalGBDT(BaseModel):
    def __init__(self, mod_name, params=None):
        self.mod_name = mod_name
        self.params = params or {}
        self.model = lgb.LGBMClassifier(**self.params)
        
    def train(self, X, y, val_data=None):
        eval_set = None
        if val_data:
            eval_set = [val_data]
        self.model.fit(X, y, eval_set=eval_set)
        
    def predict_proba(self, X, masks=None):
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, path):
        save_pickle(self, path)
        
    @classmethod
    def load(cls, path):
        return load_pickle(path)
