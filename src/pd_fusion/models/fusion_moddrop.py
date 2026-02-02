import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from pd_fusion.models.base import BaseModel

class ModalityDropoutNet(nn.Module):
    def __init__(self, modality_dims: Dict[str, int], hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        self.modality_dims = modality_dims
        self.mod_names = sorted(modality_dims.keys())
        
        # Create encoders or just input mapping?
        # Simpler: One big net, but we need to know where to mask.
        # Let's map slices.
        current_idx = 0
        self.slices = {}
        for mod in self.mod_names:
            dim = modality_dims[mod]
            self.slices[mod] = (current_idx, current_idx + dim)
            current_idx += dim
            
        input_dim = sum(modality_dims.values())
        
        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr = h
        layers.append(nn.Linear(curr, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, training_dropout: bool = False, drop_rate: float = 0.0):
        if training_dropout and self.training:
            # Randomly zero out modalities
            # Should we drop at least one? Or allow keeping all?
            # Standard ModDrop: iterate mods, drop with prob p.
            # Ensure at least one mod remains?
            
            mask = torch.ones_like(x)
            for mod in self.mod_names:
                if np.random.rand() < drop_rate:
                    start, end = self.slices[mod]
                    mask[:, start:end] = 0.0
            
            # Sanity check: if all dropped, what happens? 
            # Maybe ensure not all are dropped if we want stability, 
            # but standard dropout allows it (just noise resilience).
            x = x * mask
            
        return self.net(x)

class ModalityDropoutModel(BaseModel):
    def __init__(self, modality_dims, params):
        self.params = params
        self.modality_dims = modality_dims
        self.model = ModalityDropoutNet(modality_dims, params["hidden_dims"], params.get("dropout", 0.2))
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.criterion = nn.BCELoss()
        
    def train(self, X, y, val_data=None):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        drop_rate = self.params.get("moddrop_rate", 0.2)
        batch_size = self.params.get("batch_size", 32)
        n_samples = len(X)
        
        for epoch in range(self.params["epochs"]):
            self.model.train()
            
            # Mini-batch training
            indices = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch, training_dropout=True, drop_rate=drop_rate)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
    def predict_proba(self, X, masks=None):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            # No dropout during inference? 
            # Or use masks to zero out missing modalities explicitly?
            # Yes, if we have missing data in test, we MUST zero it out if the model expects it,
            # (though preprocess probably imputed it).
            # Better: use the user-provided mask to zero out imputed values!
            
            if masks is not None:
                # masks is typically a dict {mod: (N,)} or array logic
                # Need to map to tensor mask
                zero_mask = torch.ones_like(X_tensor)
                for mod, (start, end) in self.model.slices.items():
                    if mod in masks: # masks is dict of ndarrays
                        # mask is 1=present, 0=absent
                         m_vec = torch.FloatTensor(masks[mod]).unsqueeze(1) # [N, 1]
                         zero_mask[:, start:end] = m_vec
                X_tensor = X_tensor * zero_mask
                
            return self.model(X_tensor).numpy().flatten()
            
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    @classmethod
    def load(cls, path, input_dim, params): # Mismatch in signature handling
        # This requires modality_dims to be passed.
        # For now, simplistic load, or we pickle the whole object.
        pass 
