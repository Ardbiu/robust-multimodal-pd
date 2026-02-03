import hashlib
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]

def _hash_config(cfg: Dict) -> str:
    return hashlib.sha256(str(sorted(cfg.items())).encode()).hexdigest()[:12]

def _load_volume(path: Path, target_shape=(96, 96, 96)):
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    # Replace NaNs/Infs
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Resize to target shape
    if target_shape is not None:
        zoom = [t / s for t, s in zip(target_shape, data.shape)]
        data = ndimage.zoom(data, zoom, order=1)
    return data

def _compute_simple_features(volume: np.ndarray, hist_bins=10, grid_size=8):
    # Mask background
    mask = volume > 0
    if mask.sum() == 0:
        mask = np.ones_like(volume, dtype=bool)
    vals = volume[mask]
    mean = float(vals.mean())
    std = float(vals.std())
    vmin = float(vals.min())
    vmax = float(vals.max())
    median = float(np.median(vals))
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))

    # Histogram on clipped range
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)
    hist, _ = np.histogram(np.clip(vals, lo, hi), bins=hist_bins, range=(lo, hi), density=True)

    # Downsample grid means
    if grid_size:
        grid = ndimage.zoom(volume, (grid_size / volume.shape[0],
                                     grid_size / volume.shape[1],
                                     grid_size / volume.shape[2]), order=1)
        grid_feats = grid.flatten()
    else:
        grid_feats = np.array([])

    feats = [mean, std, vmin, vmax, median, p10, p90]
    feats.extend(hist.tolist())
    feats.extend(grid_feats.tolist())
    return np.array(feats, dtype=np.float32)

def load_simple_features(manifest_path: Path, cache_dir: Path, config: Dict) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _hash_file(manifest_path)
    cfg_hash = _hash_config(config)
    out_path = cache_dir / f"features_{manifest_hash}_{cfg_hash}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    df = pd.read_csv(manifest_path)
    hist_bins = int(config.get("hist_bins", 10))
    grid_size = int(config.get("grid_size", 8))
    target_shape = config.get("target_shape", (96, 96, 96))

    rows = []
    for _, row in df.iterrows():
        vol = _load_volume(Path(row["t1wbrain_path"]), target_shape=target_shape)
        feats = _compute_simple_features(vol, hist_bins=hist_bins, grid_size=grid_size)
        rec = {
            "subject_id": row["subject_id"],
            "session": row["session"],
            "label": int(row["label"]),
        }
        for i, v in enumerate(feats):
            rec[f"mri_feat_{i}"] = float(v)
        rows.append(rec)

    feat_df = pd.DataFrame(rows)
    feat_df.to_parquet(out_path, index=False)
    return feat_df

def load_cnn_embeddings(manifest_path: Path, cache_dir: Path, config: Dict) -> pd.DataFrame:
    """
    Loads CNN embeddings. Raises if embeddings are not precomputed.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _hash_file(manifest_path)
    cfg_hash = _hash_config(config)
    out_path = cache_dir / f"embeddings_{manifest_hash}_{cfg_hash}.parquet"
    if not out_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {out_path}. "
            "Run scripts/build_cnn3d_embeddings.py to generate them."
        )
    return pd.read_parquet(out_path)
