import logging
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from pd_fusion.utils.io import load_yaml, save_yaml, save_pickle
from pd_fusion.utils.seed import set_seed
from pd_fusion.paths import RUNS_DIR, get_run_dir
from pd_fusion.data.ppmi_loader import load_ppmi_data, create_masks_from_df
from pd_fusion.data.splits import stratified_split, get_subset_masks
from pd_fusion.training.train import train_pipeline
from pd_fusion.evaluation.evaluate import evaluate_model
from pd_fusion.evaluation.plots import (
    plot_degradation_curve, 
    plot_calibration_curve_func, 
    plot_roc_curve, 
    plot_pr_curve
)

def run_full_pipeline(config_path: str, synthetic: bool = False, overrides: dict = None):
    logger = logging.getLogger("pd_fusion")
    config = load_yaml(Path(config_path))
    
    # Apply overrides
    if overrides:
        config.update(overrides)
    
    # Load data config only if not synthetic? Or for synthetic params?
    # Usually data config provides file paths or synth params.
    # The default data config is configs/data_ppmi.yaml
    # We can assume it is present or passed in usage.
    # But cli 'run' command takes config. Config should arguably refer to data config?
    # Or we hardcode default path if not in main config.
    data_config_path = config.get("data_config", "configs/data_ppmi.yaml")
    data_config = load_yaml(Path(data_config_path))
    
    set_seed(config.get("seed", 42))
    
    if overrides and "output_dir" in overrides:
        run_id = overrides["output_dir"]
    else:
        run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    run_dir = get_run_dir(run_id)
    logger.info(f"Starting experiment: {run_id}")
    logger.info(f"Config: {config_path}")
    if overrides:
        logger.info(f"Overrides: {overrides}")
    
    # 1. Load Data
    # Support for Dev Datasets
    dataset_name = config.get("dataset", "ppmi")
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "uci_parkinsons":
        from pd_fusion.data.dev_datasets.uci_parkinsons import load_uci_parkinsons
        df, masks = load_uci_parkinsons()
        
    elif dataset_name == "uci_telemonitoring":
         from pd_fusion.data.dev_datasets.uci_telemonitoring import load_uci_telemonitoring
         df, masks = load_uci_telemonitoring()
         
    elif dataset_name == "ppmi":
        df, masks = load_ppmi_data(data_config, synthetic=synthetic)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 2. Split
    train_df, val_df, test_df = stratified_split(df)
    train_masks = get_subset_masks(masks, train_df.index)
    val_masks = get_subset_masks(masks, val_df.index)
    test_masks = get_subset_masks(masks, test_df.index)
    
    # 3. Train
    model, prep_info = train_pipeline(config, train_df, val_df, train_masks, val_masks)
    
    model.save(run_dir / "model.pt")
    save_pickle(prep_info, run_dir / "preprocess.pkl")
    
    # 4. Evaluate (Robustness Scenarios)
    # Load scenarios from eval config or use defaults
    eval_config_path = config.get("eval_config", "configs/eval_missingness.yaml")
    eval_config = load_yaml(Path(eval_config_path))
    
    # Run evaluation
    results = evaluate_model(model, test_df, test_masks, prep_info, eval_config)
    save_yaml(results, run_dir / "results.yaml")
    
    # 5. Plotting
    logger.info("Generating plots...")
    plot_degradation_curve(results, run_dir / "degradation.png")
    
    # Detailed plots for specific scenarios (e.g. Full Obs)
    # We need predictions for plotting curves. 
    # evaluate_model returns metrics. We might want raw preds for curves?
    # Re-run prediction for "full_observation" scenario to get curves.
    
    # Manual prediction for plots on Test Set (Full Observation)
    # Using helper from evaluate (simplified here)
    from pd_fusion.evaluation.evaluate import preprocess_features
    from pd_fusion.data.schema import TARGET_COL, MODALITIES
    import torch
    import numpy as np
    
    y_test = test_df[TARGET_COL].values
    
    # Helper to get preds
    def get_preds(m, df, ms, p_info):
        is_moe = isinstance(p_info, dict)
        if is_moe:
            X_d = {}
            for mod in MODALITIES:
                if mod in p_info:
                    imp, scl, fs = p_info[mod]
                    x, _, _ = preprocess_features(df, fs, imp, scl)
                    X_d[mod] = torch.FloatTensor(x)
            m_t = torch.FloatTensor(np.stack([ms[mu] for mu in MODALITIES], axis=1))
            return m.predict_proba(X_d, m_t)
        else:
            imp, scl, fs = p_info
            x, _, _ = preprocess_features(df, fs, imp, scl)
            return m.predict_proba(x, ms)

    y_prob = get_preds(model, test_df, test_masks, prep_info)
    
    plot_roc_curve(y_test, y_prob, run_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_prob, run_dir / "pr_curve.png")
    plot_calibration_curve_func(y_test, y_prob, run_dir / "calibration.png", config["model_type"])
    
    # Calibration
    if config.get("calibrate", False):
        # ... existing ...
        pass
        
    # Novelty: Mask-Conditioned Conformal Prediction
    # We calibrate the conformal wrapper on the Validation set
    # (reusing Val set is common if we don't have separate calib set, though risky for rigor. 
    #  In research, we state this limitation or split calib separately.)
    # For this skeleton, we use VAL for CP calibration.
    
    from pd_fusion.models.conformal import MaskConformalWrapper
    from pd_fusion.evaluation.evaluate import compute_risk_coverage
    from pd_fusion.evaluation.plots import plot_risk_coverage
    
    # Initialize wrapper
    cp_model = MaskConformalWrapper(model, alpha=0.1)
    
    # Ideally reuse val data logic from evaluate logic to get prepared inputs
    # We need to construct inputs for val just like we did for training
    # For Fusion:
    val_inputs = None
    if isinstance(prep_info, dict): # MoE
        val_X_d = {}
        for mod in MODALITIES:
            if mod in prep_info:
                imp, scl, fs = prep_info[mod]
                x, _, _ = preprocess_features(val_df, fs, imp, scl) # Note: val_df is raw here (not imputed?? wait. preprocess imputes)
                val_X_d[mod] = torch.FloatTensor(x)
        val_inputs = val_X_d
    else: # Late Fusion / ModDrop
        imp, scl, fs = prep_info
        x, _, _ = preprocess_features(val_df, fs, imp, scl)
        val_inputs = x
        
    # Fit conformal thresholds
    y_val = val_df[TARGET_COL].values
    cp_model.fit(val_inputs, y_val, val_masks)
    cp_model.save(run_dir / "conformal_model.pkl")
    
    # Evaluate Risk-Coverage on Test
    # Calculate probas
    # reusing test evaluation logic manually or via evaluate helper
    # We need inputs for test
    test_inputs = None
    if isinstance(prep_info, dict): # MoE
        test_X_d = {}
        for mod in MODALITIES:
            if mod in prep_info:
                imp, scl, fs = prep_info[mod]
                x, _, _ = preprocess_features(test_df, fs, imp, scl)
                test_X_d[mod] = torch.FloatTensor(x)
        test_inputs = test_X_d
    else:
        imp, scl, fs = prep_info
        x, _, _ = preprocess_features(test_df, fs, imp, scl)
        test_inputs = x
        
    # Compute base probs for curve
    # Note: cp_model.predict returns decision for fixed alpha.
    # To plot full curve, we sweep alpha or just sort by s(x).
    # compute_risk_coverage does the sweep based on confidence.
    
    # We can use the base model probs + masks for the sweep analysis
    try:
        y_prob_test = model.predict_proba(test_inputs, masks=test_masks)
    except:
        y_prob_test = model.predict_proba(test_inputs) # Handle MoE signature issue if present
        
    rc_metrics = compute_risk_coverage(test_df[TARGET_COL].values, y_prob_test, test_masks)
    plot_risk_coverage(rc_metrics, run_dir / "risk_coverage.png")
    
    logger.info(f"Experiment finished. Results saved in {run_dir}")

def run_cv_pipeline(config_path: str, k: int = 5, synthetic: bool = False, overrides: dict = None):
    logger = logging.getLogger("pd_fusion")
    config = load_yaml(Path(config_path))
    if overrides: config.update(overrides)
    
    data_config_path = config.get("data_config", "configs/data_ppmi.yaml")
    data_config = load_yaml(Path(data_config_path))
    set_seed(config.get("seed", 42))

    # 1. Load Data
    dataset_name = config.get("dataset", "ppmi")
    logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == "uci_parkinsons":
        from pd_fusion.data.dev_datasets.uci_parkinsons import load_uci_parkinsons
        df, masks = load_uci_parkinsons()
    elif dataset_name == "uci_telemonitoring":
         from pd_fusion.data.dev_datasets.uci_telemonitoring import load_uci_telemonitoring
         df, masks = load_uci_telemonitoring()
    elif dataset_name == "ppmi":
        df, masks = load_ppmi_data(data_config, synthetic=synthetic)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Output Dir
    if overrides and "output_dir" in overrides:
        run_id = overrides["output_dir"]
    else:
        run_id = f"cv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting {k}-Fold CV: {run_id}")

    # 2. CV Loop
    from pd_fusion.data.splits import get_kfold_splits, get_subset_masks
    metrics_all = []
    
    folds = get_kfold_splits(df, n_splits=k, seed=config.get("seed", 42))
    
    for i, (train_df, val_df) in enumerate(folds):
        logger.info(f"--- Fold {i+1}/{k} ---")
        
        train_masks = get_subset_masks(masks, train_df.index)
        val_masks = get_subset_masks(masks, val_df.index)
        
        # Train
        model, prep_info = train_pipeline(config, train_df, val_df, train_masks, val_masks)
        
        # Evaluate
        # Note: In CV, the validation set of the fold IS the test set for that fold.
        eval_config_path = config.get("eval_config", "configs/eval_missingness.yaml")
        eval_config = load_yaml(Path(eval_config_path))
        results = evaluate_model(model, val_df, val_masks, prep_info, eval_config)
        
        # Tag results with fold
        results["fold"] = i + 1
        metrics_all.append(results)
        
        # Save fold metrics
        save_yaml(results, run_dir / f"results_fold_{i+1}.yaml")
        
        # Save fold model (optional, skipping to save space unless requested)
        # model.save(run_dir / f"model_fold_{i+1}.pt")

    # 3. Aggregate
    logger.info("Aggregating results...")
    
    aggregated = {}
    summary_rows = []
    
    if metrics_all:
        scenarios = [k for k in metrics_all[0].keys() if k != "fold"]
        
        for scen in scenarios:
            aggregated[scen] = {}
            metric_names = metrics_all[0][scen].keys()
            
            for m in metric_names:
                values = [fold_res[scen][m] for fold_res in metrics_all]
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                aggregated[scen][m] = {"mean": mean_val, "std": std_val}
                
                summary_rows.append({
                    "scenario": scen,
                    "metric": m,
                    "mean": mean_val,
                    "std": std_val
                })
            
    # Save Aggregated
    save_yaml(aggregated, run_dir / "results_aggregated.yaml")
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary_table.csv", index=False)
    
    logger.info(f"CV Finished. Summary saved to {run_dir}")
    return aggregated
