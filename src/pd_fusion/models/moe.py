import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from pd_fusion.models.base import BaseModel

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        curr = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            curr = h
        layers.append(nn.Linear(curr, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class MoENet(nn.Module):
    def __init__(self, modality_dims: Dict[str, int], params):
        super().__init__()
        self.experts = nn.ModuleDict({
            mod: Expert(dim, params["expert_hidden_dims"]) 
            for mod, dim in modality_dims.items()
        })
        self.router = nn.Sequential(
            nn.Linear(len(modality_dims), params["router_hidden_dims"][0]),
            nn.ReLU(),
            nn.Linear(params["router_hidden_dims"][0], len(modality_dims)),
            nn.Softmax(dim=1)
        )
        
    def forward(self, modality_inputs: Dict[str, torch.Tensor], mask: torch.Tensor):
        # mask is [N, M]
        weights = self.router(mask) # [N, M]
        
        outputs = []
        mods = sorted(modality_inputs.keys())
        for i, mod in enumerate(mods):
            expert_out = self.experts[mod](modality_inputs[mod]) # [N, 1]
            outputs.append(expert_out * weights[:, i:i+1])
            
        return torch.sum(torch.stack(outputs, dim=2), dim=2) # [N, 1]

class MoEModel(BaseModel):
    def __init__(self, modality_dims, params):
        self.params = params
        self.model = MoENet(modality_dims, params)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.criterion = nn.BCELoss()
        
    def train(self, X_dict, y, mask, val_data=None):
        # X_dict: {mod: tensor}, mask: tensor [N, M]
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        for epoch in range(self.params["epochs"]):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_dict, mask)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
    def predict_proba(self, X_dict, mask=None):
        self.model.eval()
        with torch.no_grad():
            return self.model(X_dict, mask).numpy().flatten()
            
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    @classmethod
    def load(cls, path, modality_dims, params):
        instance = cls(modality_dims, params)
        instance.model.load_state_dict(torch.load(path))
        return instance
