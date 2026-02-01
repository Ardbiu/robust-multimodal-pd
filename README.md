# Robust Multimodal Fusion for Parkinson’s Diagnosis (PPMI)

This repository provides a research skeleton for Parkinson’s Disease (PD) vs. Healthy Control (HC) classification using multimodal data (Clinical, DAT-SPECT, MRI), with a focus on robustness under missing-modality conditions.

## Project Structure
- `configs/`: YAML configurations for data, models, and evaluation.
- `src/pd_fusion/`: Core logic including data loaders, models (Unimodal, Late Fusion, Masked Fusion, MoE), and training/evaluation.
- `scripts/`: Utility scripts for quickstart and full runs.
- `data/raw/`: Place your PPMI CSV files here.

## Quickstart

### 1. Installation
```bash
pip install -e .
```

### 2. Run Synthetic Experiment
To test the pipeline without real PPMI data:
```bash
bash scripts/quickstart.sh
```
This will:
- Generate synthetic data.
- Train a fusion model.
- Evaluate on missingness scenarios.
- Generate plots in `runs/`.

## Real Data Setup
1. Download PPMI CSV exports (Clinical, SBR, MRI tabular).
2. Place them in `data/raw/`.
3. Update `configs/data_ppmi.yaml` with your filenames and column mappings.
4. Run:
```bash
python -m pd_fusion.cli prepare-data --config configs/data_ppmi.yaml
```

## Models Included
- **Unimodal Baselines**: GBDT (LightGBM/XGBoost) for each modality.
- **Late Fusion**: Concatenation of features from all modalities.
- **Masked Fusion**: Fusion incorporating modality availability masks.
- **Modality Dropout**: Training with stochastic modality removal.
- **Mixture of Experts (MoE)**: Specialized experts per modality with availability-based routing.

## Evaluation
The pipeline evaluates models on:
- **Empirical Missingness**: Realistic patterns found in PPMI.
- **Stress Tests**: Single modality removal, random k-drop.
- **Calibration**: ECE and Reliability plots.
