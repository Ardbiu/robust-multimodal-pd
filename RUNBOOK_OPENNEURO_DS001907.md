# RUNBOOK: OpenNeuro ds001907 (PD vs HC)

## Environment (Engaging)
```bash
module load deprecated-modules anaconda3/2022.05-x86_64
conda activate base
pip install -e .
```

## Dataset verification
```bash
MANIFEST=/home/adixit1/IEEE-spid/data/processed/openneuro_ds001907_manifest.csv
python - <<'PY'
import pandas as pd
df = pd.read_csv("/home/adixit1/IEEE-spid/data/processed/openneuro_ds001907_manifest.csv")
print(df.shape, df.subject_id.nunique(), df.session.value_counts().to_dict(), df.label.value_counts().to_dict())
PY
```

## Simple features (CPU)
```bash
python -m pd_fusion.cli run --config configs/openneuro_ds001907_simple.yaml --k-fold 1
```

## Classical baseline (radiomics-lite)
Set `feature_config.extra_stats: true` in your data config and run:
```bash
python -m pd_fusion.cli run --config configs/openneuro_ds001907_simple.yaml --model unimodal_mri_gbdt --k-fold 5
```

## ResNet2D embeddings (transfer learning, recommended)
```bash
python scripts/build_resnet2d_embeddings.py \
  --manifest /home/adixit1/IEEE-spid/data/processed/openneuro_ds001907_manifest.csv \
  --out-dir data/processed/openneuro_ds001907/embeddings_resnet2d \
  --backbone resnet18 \
  --target-shape 160 160 160 \
  --slice-axis 2 \
  --slice-count 24 \
  --input-size 224 \
  --batch-size 32 \
  --tta 1 \
  --max-rotation-deg 5.0 \
  --max-translation 0.05 \
  --intensity-scale 0.1 \
  --intensity-shift 0.1 \
  --noise-std 0.01
```

Then use `configs/openneuro_ds001907_resnet2d.yaml` for runs.

## ResNet2D MIL embeddings (attention pooling)
```bash
python scripts/build_resnet2d_mil_embeddings.py \
  --manifest /home/adixit1/IEEE-spid/data/processed/openneuro_ds001907_manifest.csv \
  --out-dir data/processed/openneuro_ds001907/embeddings_resnet2d \
  --backbone resnet50 \
  --target-shape 160 160 160 \
  --slice-axis 2 \
  --slice-count 48 \
  --input-size 224 \
  --batch-size 16
```

Then use `configs/openneuro_ds001907_resnet2d_mil.yaml` for runs.

## Nested CV + subject-level metrics
OpenNeuro configs now enable nested calibration by default:
`nested_calibration: true` and `calibration_split: 0.2`.
Subject-level metrics are reported with the `subject_` prefix.

## Bootstrap CIs (full_observation)
```bash
python -m pd_fusion.analysis.bootstrap_ci \
  --sweep-dir runs/dual_sweep_YYYYMMDD_HHMMSS \
  --group-col subject_id \
  --n 1000
```

## CNN embeddings (GPU, optional)
```bash
python scripts/build_cnn3d_embeddings.py \
  --manifest /home/adixit1/IEEE-spid/data/processed/openneuro_ds001907_manifest.csv \
  --out-dir data/processed/openneuro_ds001907/embeddings_cnn3d \
  --target-shape 96 96 96 \
  --embedding-dim 128 \
  --epochs 10 \
  --batch-size 4
```

Then set `feature_mode: "cnn3d"` in `configs/data_openneuro_ds001907.yaml`.

## Full sweep (2×H200 per job)
```bash
python scripts/submit_dual_h200.py \
  --partition mit_normal_gpu \
  --time 05:00:00 \
  --dataset openneuro_ds001907 \
  --k-fold 5 \
  --dev-data-dir /home/adixit1/ieee-spid-data/raw_dev \
  --conda-env base \
  --module "deprecated-modules anaconda3/2022.05-x86_64" \
  --base-config configs/openneuro_ds001907_resnet2d.yaml
```

## Aggregation
```bash
SWEEP=$(ls -td runs/dual_sweep_* | head -1)
python -m pd_fusion.analysis.aggregate_results --sweep-dir $SWEEP --output $SWEEP/summary_raw.csv
```

## Figures
- `runs/<run_id>/degradation_fold1.png`
- `runs/<run_id>/roc_curve_fold1.png`
- `runs/<run_id>/pr_curve_fold1.png`
- `runs/<run_id>/calibration_fold1.png`
- `runs/<run_id>/risk_coverage_fold1.png`
