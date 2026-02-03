import argparse
from pathlib import Path
import json
import hashlib

from pd_fusion.data.openneuro_features import build_resnet2d_embeddings

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
    parser = argparse.ArgumentParser(description="Build ResNet2D embeddings for ds001907")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/openneuro_ds001907/embeddings_resnet2d",
    )
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--target-shape", type=int, nargs=3, default=[160, 160, 160])
    parser.add_argument("--slice-axis", type=int, default=2)
    parser.add_argument("--slice-count", type=int, default=24)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
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

    df = build_resnet2d_embeddings(manifest_path, out_dir, cfg)

    manifest_hash = hash_file(manifest_path)
    cfg_hash = hash_config(cfg)
    meta_path = out_dir / f"resnet2d_{manifest_hash}_{cfg_hash}.json"
    with open(meta_path, "w") as f:
        json.dump({"manifest": str(manifest_path), "config": cfg}, f, indent=2)
    print(f"Saved embeddings to {len(df)} rows in {out_dir}")


if __name__ == "__main__":
    main()
