import argparse
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]

def hash_config(cfg: dict) -> str:
    return hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12]

def load_volume(path: Path, target_shape=(96, 96, 96)):
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if target_shape is not None:
        zoom = [t / s for t, s in zip(target_shape, data.shape)]
        data = ndimage.zoom(data, zoom, order=1)
    # Normalize per-volume
    mask = data > 0
    if mask.sum() > 0:
        mean = data[mask].mean()
        std = data[mask].std() + 1e-6
        data = (data - mean) / std
    return data

class VolumeDataset(Dataset):
    def __init__(self, df, target_shape):
        self.df = df.reset_index(drop=True)
        self.target_shape = target_shape
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = load_volume(Path(row["t1wbrain_path"]), self.target_shape)
        vol = torch.from_numpy(vol).unsqueeze(0)  # [1, D, H, W]
        return vol

class Simple3DAE(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.fc = nn.Linear(32 * 12 * 12 * 12, embedding_dim)
        self.fc_dec = nn.Linear(embedding_dim, 32 * 12 * 12 * 12)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose3d(16, 8, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose3d(8, 1, 2, stride=2),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_flat = z.view(z.size(0), -1)
        emb = self.fc(z_flat)
        recon_flat = self.fc_dec(emb)
        recon = recon_flat.view(z.size(0), 32, 12, 12, 12)
        out = self.decoder(recon)
        return out, emb

def main():
    parser = argparse.ArgumentParser(description="Build CNN embeddings for ds001907")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="data/processed/openneuro_ds001907/embeddings_cnn3d")
    parser.add_argument("--target-shape", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "target_shape": args.target_shape,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    manifest_hash = hash_file(manifest_path)
    cfg_hash = hash_config(cfg)
    emb_path = out_dir / f"embeddings_{manifest_hash}_{cfg_hash}.parquet"
    meta_path = out_dir / f"embeddings_{manifest_hash}_{cfg_hash}.json"

    df = pd.read_csv(manifest_path)
    dataset = VolumeDataset(df, target_shape=tuple(args.target_shape))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DAE(embedding_dim=args.embedding_dim).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"epoch {epoch+1}/{args.epochs} loss={loss.item():.4f}")

    # Extract embeddings
    model.eval()
    emb_list = []
    with torch.no_grad():
        for idx in range(len(df)):
            vol = dataset[idx].unsqueeze(0).to(device)
            _, emb = model(vol)
            emb = emb.detach().cpu().numpy().reshape(-1)
            emb_list.append(emb)

    emb_arr = np.vstack(emb_list)
    emb_df = pd.DataFrame(emb_arr, columns=[f"mri_cnn_{i}" for i in range(emb_arr.shape[1])])
    emb_df["subject_id"] = df["subject_id"].values
    emb_df["session"] = df["session"].values
    emb_df["label"] = df["label"].values

    emb_df.to_parquet(emb_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({"manifest": str(manifest_path), "config": cfg}, f, indent=2)
    print(f"Saved embeddings to {emb_path}")

if __name__ == "__main__":
    main()
