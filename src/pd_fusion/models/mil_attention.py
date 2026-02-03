import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pd_fusion.models.base import BaseModel


class MILAttentionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, attn_dim: int, dropout: float = 0.3):
        super().__init__()
        self.instance = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, L, D]
        h = self.instance(x)  # [B, L, H]
        scores = self.attn(h).squeeze(-1)  # [B, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)  # [B, L]
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)  # [B, H]
        return self.classifier(pooled).squeeze(-1)


def _pad_bags(bags):
    max_len = max(b.shape[0] for b in bags)
    feat_dim = bags[0].shape[1]
    X = np.zeros((len(bags), max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((len(bags), max_len), dtype=np.float32)
    for i, bag in enumerate(bags):
        length = bag.shape[0]
        X[i, :length, :] = bag
        mask[i, :length] = 1.0
    return X, mask


class MilAttentionModel(BaseModel):
    def __init__(self, input_dim: int, params: dict):
        self.params = params or {}
        hidden_dim = int(self.params.get("hidden_dim", 128))
        attn_dim = int(self.params.get("attn_dim", 64))
        dropout = float(self.params.get("dropout", 0.3))
        self.missing_prob = float(self.params.get("missing_prob", 0.5))

        self.model = MILAttentionNet(input_dim, hidden_dim, attn_dim, dropout)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(self.params.get("lr", 1e-3)),
            weight_decay=float(self.params.get("weight_decay", 0.0)),
        )
        self.criterion = nn.BCELoss()

    def train(self, bags, y, val_data=None):
        X, mask = _pad_bags(bags)
        X_tensor = torch.FloatTensor(X)
        mask_tensor = torch.FloatTensor(mask)
        y_tensor = torch.FloatTensor(y)

        dataset = TensorDataset(X_tensor, mask_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=int(self.params.get("batch_size", 16)),
            shuffle=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        for _ in range(int(self.params.get("epochs", 30))):
            self.model.train()
            for xb, mb, yb in loader:
                xb = xb.to(device)
                mb = mb.to(device)
                yb = yb.to(device)
                preds = self.model(xb, mb)
                loss = self.criterion(preds, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, bags, masks=None):
        # masks may be dict with "mri" indicating missing samples
        mri_mask = None
        if isinstance(masks, dict) and "mri" in masks:
            mri_mask = masks["mri"]

        probs = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            for i, bag in enumerate(bags):
                if bag is None or (mri_mask is not None and mri_mask[i] == 0):
                    probs.append(self.missing_prob)
                    continue
                X, mask = _pad_bags([bag])
                xb = torch.FloatTensor(X).to(device)
                mb = torch.FloatTensor(mask).to(device)
                pred = self.model(xb, mb).cpu().numpy().flatten()[0]
                probs.append(float(pred))
        return np.array(probs)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path, input_dim, params):
        instance = cls(input_dim, params)
        instance.model.load_state_dict(torch.load(path))
        return instance
