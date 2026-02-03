import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pd_fusion.models.base import BaseModel
from pd_fusion.models.mil_attention import MILAttentionNet
from pd_fusion.data.openneuro_features import (
    _load_volume,
    _normalize_volume_for_resnet,
    _select_slices,
    _apply_affine_2d,
    _build_resnet_backbone,
)


class MilAttentionFineTuneModel(BaseModel):
    """
    End-to-end MIL with a ResNet2D backbone over slices + attention pooling head.
    Bags can be file paths (str) or precomputed slice arrays (np.ndarray).
    """
    def __init__(self, params: dict):
        self.params = params or {}
        self.backbone_name = self.params.get("backbone", "resnet50")
        self.target_shape = tuple(self.params.get("target_shape", (160, 160, 160)))
        self.slice_axes = self.params.get("slice_axes", None)
        self.slice_counts = self.params.get("slice_counts", None)
        self.slice_axis = int(self.params.get("slice_axis", 2))
        self.slice_count = int(self.params.get("slice_count", 48))
        self.input_size = int(self.params.get("input_size", 224))
        self.slice_batch_size = int(self.params.get("slice_batch_size", 16))
        self.bag_batch_size = int(self.params.get("batch_size", 4))
        self.tta = int(self.params.get("tta", 1))
        self.max_rotation = float(self.params.get("max_rotation_deg", 5.0))
        self.max_translation = float(self.params.get("max_translation", 0.05))
        self.intensity_scale = float(self.params.get("intensity_scale", 0.1))
        self.intensity_shift = float(self.params.get("intensity_shift", 0.1))
        self.noise_std = float(self.params.get("noise_std", 0.01))
        self.missing_prob = float(self.params.get("missing_prob", 0.5))
        self.freeze_backbone_epochs = int(self.params.get("freeze_backbone_epochs", 2))

        hidden_dim = int(self.params.get("hidden_dim", 256))
        attn_dim = int(self.params.get("attn_dim", 128))
        dropout = float(self.params.get("dropout", 0.2))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained = bool(self.params.get("pretrained", True))
        self.backbone, self.emb_dim, self.weights = _build_resnet_backbone(self.backbone_name, pretrained=pretrained)
        self.backbone = self.backbone.to(self.device).float()
        gated = bool(self.params.get("gated", False))
        self.attn = MILAttentionNet(self.emb_dim, hidden_dim, attn_dim, dropout, gated=gated).to(self.device).float()

        if hasattr(self.weights, "meta"):
            mean_vals = self.weights.meta.get("mean", [0.5, 0.5, 0.5])
            std_vals = self.weights.meta.get("std", [0.5, 0.5, 0.5])
        else:
            mean_vals = [0.5, 0.5, 0.5]
            std_vals = [0.5, 0.5, 0.5]
        self.mean = torch.tensor(mean_vals, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(std_vals, dtype=torch.float32).view(1, 3, 1, 1).to(self.device)

        lr_backbone = float(self.params.get("lr_backbone", 1e-4))
        lr_head = float(self.params.get("lr", 3e-4))
        weight_decay = float(self.params.get("weight_decay", 1e-3))
        self.optimizer = optim.Adam(
            [
                {"params": self.backbone.parameters(), "lr": lr_backbone},
                {"params": self.attn.parameters(), "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )
        self.criterion = nn.BCELoss(reduction="none")
        self.pos_weight = None
        if self.params.get("class_weight") == "balanced":
            self.pos_weight = None
        elif self.params.get("pos_weight") is not None:
            self.pos_weight = float(self.params.get("pos_weight"))

    def _set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def _select_slices_multi(self, volume: np.ndarray) -> np.ndarray:
        if self.slice_axes and self.slice_counts:
            slices_list = []
            for axis, count in zip(self.slice_axes, self.slice_counts):
                slices_list.append(_select_slices(volume, int(axis), int(count)))
            return np.concatenate(slices_list, axis=0)
        return _select_slices(volume, self.slice_axis, self.slice_count)

    def _augment_slices(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.tta <= 1:
            return slices
        aug = slices.copy()
        angle = rng.uniform(-self.max_rotation, self.max_rotation)
        translate = rng.uniform(-self.max_translation, self.max_translation, size=2)
        translate = translate * np.array([aug.shape[1], aug.shape[2]])
        for i in range(aug.shape[0]):
            aug[i] = _apply_affine_2d(aug[i], angle, translate)
        scale = 1.0 + rng.uniform(-self.intensity_scale, self.intensity_scale)
        shift = rng.uniform(-self.intensity_shift, self.intensity_shift)
        aug = aug * scale + shift
        if self.noise_std > 0:
            aug = aug + rng.normal(0.0, self.noise_std, size=aug.shape)
        aug = np.clip(aug, 0.0, 1.0)
        return aug.astype(np.float32, copy=False)

    def _load_bag(self, bag, train: bool = False) -> np.ndarray:
        if bag is None:
            return None
        if isinstance(bag, np.ndarray):
            return bag.astype(np.float32, copy=False)
        vol = _load_volume(bag, target_shape=self.target_shape)
        vol = _normalize_volume_for_resnet(vol)
        slices = self._select_slices_multi(vol)
        if train and self.tta > 1:
            rng_seed = abs(hash(str(bag))) % (2**32)
            rng = np.random.default_rng(rng_seed)
            slices = self._augment_slices(slices, rng)
        return slices.astype(np.float32, copy=False)

    def _slices_to_tensor(self, slices: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(slices).unsqueeze(1).float()
        t = F.interpolate(t, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        t = t.repeat(1, 3, 1, 1)
        t = (t - self.mean) / self.std
        return t

    def _forward_bags(self, bags, train: bool = False):
        feats_list = []
        for bag in bags:
            if bag is None:
                feats_list.append(None)
                continue
            slices = self._load_bag(bag, train=train)
            if slices is None:
                feats_list.append(None)
                continue
            slice_tensor = self._slices_to_tensor(slices).to(self.device)
            feats = []
            for i in range(0, slice_tensor.size(0), self.slice_batch_size):
                batch = slice_tensor[i:i + self.slice_batch_size]
                out = self.backbone(batch)
                feats.append(out)
            feats_list.append(torch.cat(feats, dim=0))

        max_len = max(f.shape[0] for f in feats_list if f is not None)
        X = torch.zeros((len(feats_list), max_len, self.emb_dim), device=self.device)
        mask = torch.zeros((len(feats_list), max_len), device=self.device)
        for i, f in enumerate(feats_list):
            if f is None:
                continue
            L = f.shape[0]
            X[i, :L, :] = f
            mask[i, :L] = 1.0
        return X, mask

    def train(self, bags, y, val_data=None):
        y = np.asarray(y, dtype=np.float32)
        n = len(bags)
        epochs = int(self.params.get("epochs", 20))
        max_grad_norm = self.params.get("max_grad_norm")
        patience = int(self.params.get("early_stopping_patience", 0))
        best_auc = -1.0
        best_state = None
        bad_epochs = 0

        if self.pos_weight is None and self.params.get("class_weight") == "balanced":
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            if pos > 0:
                self.pos_weight = neg / pos

        for epoch in range(epochs):
            self.backbone.train()
            self.attn.train()
            self._set_backbone_trainable(epoch >= self.freeze_backbone_epochs)

            idxs = np.random.permutation(n)
            for start in range(0, n, self.bag_batch_size):
                batch_idx = idxs[start:start + self.bag_batch_size]
                batch_bags = [bags[i] for i in batch_idx]
                batch_y = torch.from_numpy(y[batch_idx]).to(self.device)
                X, mask = self._forward_bags(batch_bags, train=True)
                preds = self.attn(X, mask)
                loss_vec = self.criterion(preds, batch_y)
                if self.pos_weight is not None:
                    weights = torch.where(batch_y >= 0.5, torch.tensor(self.pos_weight, device=self.device), torch.tensor(1.0, device=self.device))
                    loss = (loss_vec * weights).mean()
                else:
                    loss = loss_vec.mean()
                self.optimizer.zero_grad()
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(list(self.backbone.parameters()) + list(self.attn.parameters()), float(max_grad_norm))
                self.optimizer.step()

            if val_data is not None and patience > 0:
                from sklearn.metrics import roc_auc_score
                X_val_bags, y_val = val_data
                y_prob = self.predict_proba(X_val_bags)
                try:
                    auc = float(roc_auc_score(y_val, y_prob))
                except Exception:
                    auc = -1.0
                if auc > best_auc:
                    best_auc = auc
                    best_state = {
                        "backbone": self.backbone.state_dict(),
                        "attn": self.attn.state_dict(),
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break

        if best_state is not None:
            self.backbone.load_state_dict(best_state["backbone"])
            self.attn.load_state_dict(best_state["attn"])

    def predict_proba(self, bags, masks=None):
        mri_mask = None
        if isinstance(masks, dict) and "mri" in masks:
            mri_mask = masks["mri"]
        probs = []
        self.backbone.eval()
        self.attn.eval()
        with torch.no_grad():
            for i, bag in enumerate(bags):
                if bag is None or (mri_mask is not None and mri_mask[i] == 0):
                    probs.append(self.missing_prob)
                    continue
                X, mask = self._forward_bags([bag], train=False)
                pred = self.attn(X, mask).cpu().numpy().flatten()[0]
                probs.append(float(pred))
        return np.array(probs)

    def save(self, path):
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "attn": self.attn.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, params):
        instance = cls(params)
        state = torch.load(path, map_location="cpu")
        instance.backbone.load_state_dict(state["backbone"])
        instance.attn.load_state_dict(state["attn"])
        return instance
