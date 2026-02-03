import torch
import torch.nn as nn
import torch.optim as optim
from pd_fusion.models.base import BaseModel

class LateFusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class LateFusionModel(BaseModel):
    def __init__(self, input_dim, params):
        self.params = params
        self.model = LateFusionNet(input_dim, params["hidden_dims"], params.get("dropout", 0.2))
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=params["lr"],
            weight_decay=params.get("weight_decay", 0.0),
        )
        self.criterion = nn.BCELoss()
        
    def train(self, X, y, val_data=None):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
        for epoch in range(self.params["epochs"]):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
            
    def predict_proba(self, X, masks=None):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            return self.model(X_tensor).numpy().flatten()
            
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    @classmethod
    def load(cls, path, input_dim, params):
        instance = cls(input_dim, params)
        instance.model.load_state_dict(torch.load(path))
        return instance
