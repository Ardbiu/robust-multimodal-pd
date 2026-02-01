import pytest
import numpy as np
import pandas as pd
from pd_fusion.data.ppmi_loader import generate_synthetic_data
from pd_fusion.data.missingness import apply_missingness_scenario
from pd_fusion.data.splits import stratified_split
from pd_fusion.utils.metrics import compute_metrics

def test_synthetic_data_generation():
    config = {
        "num_samples": 100,
        "missing_rates": [0.1, 0.2, 0.3],
        "clinical_dim": 5,
        "datspect_dim": 5,
        "mri_dim": 5
    }
    df, masks = generate_synthetic_data(config)
    assert len(df) == 100
    assert "diagnosis" in df.columns
    assert len(masks) == 3

def test_missingness_application():
    df = pd.DataFrame({"a": [1, 2, 3]})
    mask = {"m1": np.array([1, 1, 1])}
    scenario = {"drop_modalities": ["m1"]}
    new_mask = apply_missingness_scenario(df, scenario, mask)
    assert np.all(new_mask["m1"] == 0)

def test_stratified_split():
    df = pd.DataFrame({
        "patno": range(100),
        "diagnosis": [0]*50 + [1]*50
    })
    train, val, test = stratified_split(df, test_size=0.2, val_size=0.1)
    assert len(train) == 70
    assert len(val) == 10
    assert len(test) == 20
    assert train["diagnosis"].sum() == 35 # Stratified

def test_metrics():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    m = compute_metrics(y_true, y_prob)
    assert m["roc_auc"] == 1.0
    assert m["ece"] >= 0
活跃的
