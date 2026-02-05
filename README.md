# Robust Multimodal Fusion for Parkinson’s Diagnosis (PPMI)

This repository provides a research codebase for Parkinson’s Disease (PD) vs. Healthy Control (HC) classification, focusing on robustness under missing-modality conditions. It supports synthetic data prototyping and real PPMI data integration.

## Quickstart (Synthetic Data)

Run the full end-to-end pipeline on synthetic data to verify setup:
```bash
./scripts/quickstart.sh
```
Outputs will be saved in `runs/run_<timestamp>/`, including:
- `model.pt`: Trained model.
- `results.yaml`: Metrics for all scenarios.
- `degradation.png`, `roc_curve.png`, `calibration.png`: Evaluation plots.
- `risk_coverage.png`: **[NEW]** Clinical utility curve (Risk vs Coverage).

## Real Data Workflow

### 1. Place Data
Download PPMI CSV exports (Clinical, SBR, MRI) and place them in `data/raw/`.

### 2. Configure Columns
Edit `configs/ppmi_columns.yaml` to map your raw CSV column headers to the canonical names used by the pipeline (e.g., mapping "NP3TOT" to "updrs_iii").

### 3. Validate & Merge
Run the data validation tool to map, merge, and check your data:
```bash
python -m pd_fusion.cli validate-data --config configs/data_ppmi.yaml
```
This produces `data/processed/ppmi_merged.parquet` and prints missingness statistics.

### 4. Run Experiments
Execute the research pipeline using a specific configuration:
```bash
python -m pd_fusion.cli run --config configs/model_fusion.yaml
```

## PPMI Study Data Pipeline (Tabular)

This pipeline ingests PPMI “Study Data” tables (CSV) and builds baseline/visit-level datasets.

### 1. Place raw study-data CSVs (or ZIPs)
Put PPMI study-data downloads under:
```
/home/adixit1/IEEE-spid/data/raw_ppmi/study_data/
```
If ZIPs are present, the build script will extract them automatically.

### 2. Build processed datasets
```bash
python scripts/ppmi_build_dataset.py --config configs/ppmi_studydata.yaml
```
Outputs are written to:
```
/home/adixit1/IEEE-spid/data/processed/ppmi/
```
Key artifacts:
- `ppmi_subject_baseline.csv`
- `ppmi_visit_level.csv`
- `ppmi_feature_schema.json`
- `ppmi_splits_seed{SEED}.json`
- `ppmi_manifest.md`

### 3. Train tabular baselines + ablations
```bash
python scripts/ppmi_train_tabular.py --config configs/ppmi_studydata.yaml
```
This writes results to:
```
/home/adixit1/IEEE-spid/runs/ppmi_tabular_<timestamp>/
```

### 4. Generate summary/ranking table
```bash
python scripts/ppmi_eval_report.py --config configs/ppmi_studydata.yaml --out_dir /home/adixit1/IEEE-spid/runs/ppmi_tabular_<timestamp>
```

### Smoke test
```bash
scripts/ppmi_smoke.sh configs/ppmi_studydata.yaml
```

## Meaningful PPMI baselines

We provide a focused baseline suite for PD vs HC that separates motor vs non‑motor contributions and imaging‑only signals.

Settings:
- `full_clinical`: all numeric features (current baseline).
- `no_motor_exam`: drops MDS‑UPDRS + NHY + tremor/rigidity/bradykinesia columns.
- `non_motor_only`: cognition, sleep, mood/anxiety/depression, UPSIT/olfaction.
- `datsbr_only`: DAT SBR‑related columns.
- `freesurfer_only`: MRI‑derived cortical thickness/volume/area.
- `fusion_nonmotor_imaging`: non‑motor + DAT SBR + MRI‑derived.

Run it:
```bash
python scripts/ppmi_meaningful_suite.py \
  --input-csv /home/adixit1/IEEE-spid/data/processed/ppmi/ppmi_subject_baseline.csv
```

Outputs go to:
```
/home/adixit1/IEEE-spid/runs/ppmi_meaningful_suite_<timestamp>/
```
with summary tables, per‑fold metrics, feature importances, permutation tests, and a ROC‑AUC bar plot.

## PPMI Imaging Upgrade Suite

This suite audits imaging features, builds curated ROI subsets, applies covariate adjustment/harmonization, and re‑runs imaging‑only + fusion models. It also supports harder longitudinal endpoints (conversion/progression).

Config:
```
configs/ppmi_imaging_upgrade.yaml
```

Run:
```bash
python scripts/ppmi_imaging_upgrade.py --config configs/ppmi_imaging_upgrade.yaml
```

Key outputs (per run directory):
- `summary_mean.csv`, `per_fold_metrics.csv`, `feature_importance.csv`
- `imaging_missingness_per_feature.csv`, `imaging_missingness_per_subject.csv`
- `predictions.csv`, `roc_curves.png`, `calibration_curves.png`
- `paired_tests.json`, `univariate_top.csv`

## Models & Architectures
- **Fusion ModDrop**: Late fusion network with randomized modality dropout during training.
- **MoE**: Mixture of Experts with a router conditioned on the missingness mask.
- **Unimodal Baselines**: Gradient Boosted Trees for individual modalities.
- **Calibration**: Isotonic regression or Temperature scaling wrappers.
- **Conformal Prediction**: Mask-Conditioned abstention logic.

## Evaluation Scenarios
The codebase evaluates robustness against:
- **Empirical Missingness**: Masks sampled from the test set.
- **Stress Tests**: Complete removal of key modalities (e.g., "missing_dat").
- **Random k-Drop**: Randomly dropping $k$ modalities to test degradation.

## Directory Structure
- `configs/`: YAML configurations.
- `src/pd_fusion/`: Core package.
  - `features/`: Extraction logic for Clinical, DAT, MRI.
  - `models/`: PyTorch and Sklearn model implementations.
  - `experiments/`: Main pipeline orchestration.
- `runs/`: Experiment artifacts (plots, metrics, checkpoints).

## Dev Benchmark Datasets (Open Access)
Supported dev datasets (no credentials required):
- UCI Parkinson’s (classification)
- UCI Telemonitoring (binary severity proxy)
- OpenNeuro: ds004471, ds004392, ds001907 (metadata + MRI file-count proxies)

Download dev datasets:
```bash
python -m pd_fusion.cli download-dev --dataset all
```
For metadata-only OpenNeuro downloads:
```bash
python -m pd_fusion.cli download-dev --dataset openneuro --openneuro-metadata-only
```
If label columns differ, update `configs/openneuro_labels.yaml`.
Override dev data location with `PD_FUSION_DEV_DATA_DIR` (e.g., for shared storage).
