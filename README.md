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
