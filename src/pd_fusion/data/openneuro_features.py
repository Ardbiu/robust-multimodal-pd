import hashlib
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
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
    import nibabel as nib
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    # Replace NaNs/Infs
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Resize to target shape
    if target_shape is not None:
        zoom = [t / s for t, s in zip(target_shape, data.shape)]
        data = ndimage.zoom(data, zoom, order=1)
    return data

def _compute_simple_features(volume: np.ndarray, hist_bins=10, grid_size=8, extra_stats: bool = False):
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
    if extra_stats:
        from scipy.stats import skew, kurtosis
        sk = float(np.nan_to_num(skew(vals), nan=0.0))
        kt = float(np.nan_to_num(kurtosis(vals), nan=0.0))
        # Histogram entropy
        h = hist + 1e-12
        ent = float(-(h * np.log(h)).sum())
        feats.extend([sk, kt, ent])
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
    extra_stats = bool(config.get("extra_stats", False))

    rows = []
    for _, row in df.iterrows():
        vol = _load_volume(Path(row["t1wbrain_path"]), target_shape=target_shape)
        feats = _compute_simple_features(vol, hist_bins=hist_bins, grid_size=grid_size, extra_stats=extra_stats)
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

def _normalize_volume_for_resnet(volume: np.ndarray) -> np.ndarray:
    mask = volume > 0
    if mask.sum() > 0:
        vals = volume[mask]
        lo = np.percentile(vals, 1)
        hi = np.percentile(vals, 99)
    else:
        lo = float(np.min(volume))
        hi = float(np.max(volume))
    vol = np.clip(volume, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-6)
    return vol.astype(np.float32)

def _select_slices(volume: np.ndarray, axis: int, slice_count: int) -> np.ndarray:
    axis_len = volume.shape[axis]
    other_axes = tuple(i for i in range(3) if i != axis)
    nonzero = np.any(volume > 0, axis=other_axes)
    idxs = np.where(nonzero)[0]
    if len(idxs) == 0:
        idxs = np.arange(axis_len)
    lo, hi = int(idxs[0]), int(idxs[-1])
    if slice_count > (hi - lo + 1):
        slice_count = hi - lo + 1
    indices = np.linspace(lo, hi, slice_count).astype(int)
    if axis == 0:
        slices = volume[indices, :, :]
    elif axis == 1:
        slices = volume[:, indices, :].transpose(1, 0, 2)
    else:
        slices = volume[:, :, indices].transpose(2, 0, 1)
    return slices

def _build_resnet_backbone(backbone: str, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
    if backbone == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    emb_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, emb_dim, weights

def _apply_affine_2d(slice_2d: np.ndarray, angle_deg: float, translate: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = np.array(slice_2d.shape) / 2.0
    offset = center - rot @ center + translate
    return ndimage.affine_transform(
        slice_2d,
        rot,
        offset=offset,
        order=1,
        mode="constant",
        cval=0.0,
    )

def build_resnet2d_embeddings(manifest_path: Path, cache_dir: Path, config: Dict) -> pd.DataFrame:
    """
    Builds 2D slice-based embeddings using a pretrained ResNet backbone.
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _hash_file(manifest_path)
    cfg_hash = _hash_config(config)
    out_path = cache_dir / f"resnet2d_{manifest_hash}_{cfg_hash}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    df = pd.read_csv(manifest_path)
    backbone = config.get("backbone", "resnet18")
    target_shape = tuple(config.get("target_shape", (160, 160, 160)))
    slice_axis = int(config.get("slice_axis", 2))
    slice_count = int(config.get("slice_count", 24))
    input_size = int(config.get("input_size", 224))
    batch_size = int(config.get("batch_size", 32))
    tta = int(config.get("tta", 1))
    max_rotation = float(config.get("max_rotation_deg", 5.0))
    max_translation = float(config.get("max_translation", 0.05))
    intensity_scale = float(config.get("intensity_scale", 0.1))
    intensity_shift = float(config.get("intensity_shift", 0.1))
    noise_std = float(config.get("noise_std", 0.01))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, emb_dim, weights = _build_resnet_backbone(backbone)
    model = model.to(device)
    model.eval()

    if hasattr(weights, "meta"):
        mean_vals = weights.meta.get("mean", [0.5, 0.5, 0.5])
        std_vals = weights.meta.get("std", [0.5, 0.5, 0.5])
    else:
        mean_vals = [0.5, 0.5, 0.5]
        std_vals = [0.5, 0.5, 0.5]
    mean = torch.tensor(mean_vals).view(1, 3, 1, 1).to(device)
    std = torch.tensor(std_vals).view(1, 3, 1, 1).to(device)

    rows = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="ResNet2D embeddings"):
            vol = _load_volume(Path(row["t1wbrain_path"]), target_shape=target_shape)
            vol = _normalize_volume_for_resnet(vol)
            slices = _select_slices(vol, slice_axis, slice_count)

            rng_seed = abs(hash(str(row.get("subject_id", "")))) % (2**32)
            rng = np.random.default_rng(rng_seed)

            emb_accum = None
            for _ in range(max(1, tta)):
                aug_slices = slices.copy()
                if tta > 1:
                    angle = rng.uniform(-max_rotation, max_rotation)
                    translate = rng.uniform(-max_translation, max_translation, size=2)
                    translate = translate * np.array([aug_slices.shape[1], aug_slices.shape[2]])
                    for i in range(aug_slices.shape[0]):
                        aug_slices[i] = _apply_affine_2d(aug_slices[i], angle, translate)
                    scale = 1.0 + rng.uniform(-intensity_scale, intensity_scale)
                    shift = rng.uniform(-intensity_shift, intensity_shift)
                    aug_slices = aug_slices * scale + shift
                    if noise_std > 0:
                        aug_slices = aug_slices + rng.normal(0.0, noise_std, size=aug_slices.shape)
                    aug_slices = np.clip(aug_slices, 0.0, 1.0)

                slice_tensor = torch.from_numpy(aug_slices).unsqueeze(1)  # [N,1,H,W]
                slice_tensor = F.interpolate(
                    slice_tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
                )
                slice_tensor = slice_tensor.repeat(1, 3, 1, 1).to(device)
                slice_tensor = (slice_tensor - mean) / std

                feats = []
                for i in range(0, slice_tensor.size(0), batch_size):
                    batch = slice_tensor[i:i + batch_size]
                    out = model(batch)
                    feats.append(out.detach().cpu())
                emb = torch.cat(feats, dim=0).mean(dim=0).numpy()
                emb_accum = emb if emb_accum is None else (emb_accum + emb)

            emb = emb_accum / max(1, tta)

            rec = {
                "subject_id": row["subject_id"],
                "session": row["session"],
                "label": int(row["label"]),
            }
            for i, v in enumerate(emb.tolist()):
                rec[f"mri_resnet_{i}"] = float(v)
            rows.append(rec)

    emb_df = pd.DataFrame(rows)
    emb_df.to_parquet(out_path, index=False)
    return emb_df

def load_resnet2d_embeddings(manifest_path: Path, cache_dir: Path, config: Dict) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _hash_file(manifest_path)
    cfg_hash = _hash_config(config)
    out_path = cache_dir / f"resnet2d_{manifest_hash}_{cfg_hash}.parquet"
    if not out_path.exists():
        raise FileNotFoundError(
            f"ResNet2D embeddings not found at {out_path}. "
            "Run scripts/build_resnet2d_embeddings.py to generate them."
        )
    return pd.read_parquet(out_path)

def load_resnet2d_mil_embeddings(manifest_path: Path, cache_dir: Path, config: Dict) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _hash_file(manifest_path)
    cfg_hash = _hash_config(config)
    out_path = cache_dir / f"resnet2d_mil_{manifest_hash}_{cfg_hash}.npz"
    if not out_path.exists():
        raise FileNotFoundError(
            f"ResNet2D MIL embeddings not found at {out_path}. "
            "Run scripts/build_resnet2d_mil_embeddings.py to generate them."
        )
    data = np.load(out_path, allow_pickle=True)
    emb = data["embeddings"]
    df = pd.DataFrame({
        "subject_id": data["subject_id"],
        "session": data["session"],
        "label": data["label"],
    })
    df["mri_mil"] = list(emb)
    return df
