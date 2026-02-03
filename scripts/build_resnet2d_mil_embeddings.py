import argparse
from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pd_fusion.data.openneuro_features import (
    _load_volume,
    _normalize_volume_for_resnet,
    _select_slices,
    _build_resnet_backbone,
    _apply_affine_2d,
)

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


def main():
    parser = argparse.ArgumentParser(description="Build ResNet2D MIL embeddings for ds001907")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/openneuro_ds001907/embeddings_resnet2d",
    )
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--target-shape", type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--slice-count", type=int, default=48)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tta", type=int, default=1)
    parser.add_argument("--max-rotation-deg", type=float, default=5.0)
    parser.add_argument("--max-translation", type=float, default=0.05)
    parser.add_argument("--intensity-scale", type=float, default=0.1)
    parser.add_argument("--intensity-shift", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.01)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "backbone": args.backbone,
        "target_shape": args.target_shape,
        "slice_axis": args.slice_axis,
        "slice_count": args.slice_count,
        "input_size": args.input_size,
        "batch_size": args.batch_size,
        "tta": args.tta,
        "max_rotation_deg": args.max_rotation_deg,
        "max_translation": args.max_translation,
        "intensity_scale": args.intensity_scale,
        "intensity_shift": args.intensity_shift,
        "noise_std": args.noise_std,
    }

    manifest_hash = hash_file(manifest_path)
    cfg_hash = hash_config(cfg)
    out_path = out_dir / f"resnet2d_mil_{manifest_hash}_{cfg_hash}.npz"
    meta_path = out_dir / f"resnet2d_mil_{manifest_hash}_{cfg_hash}.json"

    df = pd.read_csv(manifest_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, emb_dim, weights = _build_resnet_backbone(args.backbone)
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

    embeddings = []
    for _, row in df.iterrows():
        vol = _load_volume(Path(row["t1wbrain_path"]), target_shape=tuple(args.target_shape))
        vol = _normalize_volume_for_resnet(vol)
        slices = _select_slices(vol, args.slice_axis, args.slice_count)

        rng_seed = abs(hash(str(row.get("subject_id", "")))) % (2**32)
        rng = np.random.default_rng(rng_seed)

        emb_accum = None
        for _ in range(max(1, args.tta)):
            aug_slices = slices.copy()
            if args.tta > 1:
                angle = rng.uniform(-args.max_rotation_deg, args.max_rotation_deg)
                translate = rng.uniform(-args.max_translation, args.max_translation, size=2)
                translate = translate * np.array([aug_slices.shape[1], aug_slices.shape[2]])
                for i in range(aug_slices.shape[0]):
                    aug_slices[i] = _apply_affine_2d(aug_slices[i], angle, translate)
                scale = 1.0 + rng.uniform(-args.intensity_scale, args.intensity_scale)
                shift = rng.uniform(-args.intensity_shift, args.intensity_shift)
                aug_slices = aug_slices * scale + shift
                if args.noise_std > 0:
                    aug_slices = aug_slices + rng.normal(0.0, args.noise_std, size=aug_slices.shape)
                aug_slices = np.clip(aug_slices, 0.0, 1.0)

            slice_tensor = torch.from_numpy(aug_slices).unsqueeze(1)  # [N,1,H,W]
            slice_tensor = F.interpolate(
                slice_tensor, size=(args.input_size, args.input_size), mode="bilinear", align_corners=False
            )
            slice_tensor = slice_tensor.repeat(1, 3, 1, 1).to(device)
            slice_tensor = (slice_tensor - mean) / std

            feats = []
            with torch.no_grad():
                for i in range(0, slice_tensor.size(0), args.batch_size):
                    batch = slice_tensor[i:i + args.batch_size]
                    out = model(batch)
                    feats.append(out.detach().cpu())
            emb = torch.cat(feats, dim=0).numpy()  # [slice_count, emb_dim]
            emb_accum = emb if emb_accum is None else (emb_accum + emb)

        emb = emb_accum / max(1, args.tta)
        embeddings.append(emb)

    emb_arr = np.stack(embeddings, axis=0).astype(np.float32)

    np.savez_compressed(
        out_path,
        embeddings=emb_arr,
        subject_id=df["subject_id"].values,
        session=df["session"].values,
        label=df["label"].values,
    )

    with open(meta_path, "w") as f:
        json.dump({"manifest": str(manifest_path), "config": cfg}, f, indent=2)

    print(f"Saved MIL embeddings to {out_path}")


if __name__ == "__main__":
    main()
