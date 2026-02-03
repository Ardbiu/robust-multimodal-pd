import logging
import datetime
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from pd_fusion.utils.io import load_yaml, save_yaml, save_pickle
from pd_fusion.utils.seed import set_seed
from pd_fusion.paths import RUNS_DIR, get_run_dir, ROOT_DIR
from pd_fusion.data.ppmi_loader import load_ppmi_data, create_masks_from_df
from pd_fusion.data.splits import stratified_split, get_subset_masks
from pd_fusion.training.train import train_pipeline
from pd_fusion.evaluation.evaluate import evaluate_model
from pd_fusion.data.missingness import get_modality_mask_matrix
from pd_fusion.data.feature_utils import apply_masks_to_matrix
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
         
    elif dataset_name == "openneuro_ds001907":
        from pd_fusion.data.openneuro_ds001907 import load_openneuro_ds001907
        df, masks = load_openneuro_ds001907(data_config)

    elif dataset_name.startswith("openneuro_") or dataset_name in ["ds004471", "ds004392", "ds001907"]:
        from pd_fusion.data.dev_datasets.openneuro import load_openneuro_dataset
        accession = dataset_name.replace("openneuro_", "")
        df, masks = load_openneuro_dataset(accession)

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

    _save_run_provenance(
        run_dir=run_dir,
        config=config,
        eval_config=eval_config,
        dataset_name=dataset_name,
        synthetic=synthetic,
        overrides=overrides,
    )
    
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
            mods_used = list(p_info.keys())
            for mod in mods_used:
                imp, scl, fs = p_info[mod]
                x, _, _ = preprocess_features(df, fs, imp, scl)
                if mod in ms:
                    x = x * ms[mod].reshape(-1, 1)
                X_d[mod] = torch.FloatTensor(x)
            m_t = torch.FloatTensor(np.stack([ms[mu] for mu in mods_used], axis=1))
            return m.predict_proba(X_d, m_t)
        else:
            imp, scl, fs = p_info
            x, _, _ = preprocess_features(df, fs, imp, scl)
            x = apply_masks_to_matrix(x, ms, fs)
            if hasattr(m, "mask_dim"):
                mask_mat = get_modality_mask_matrix(ms)
                return m.predict_proba(x, mask_mat)
            return m.predict_proba(x, ms)

    y_prob = get_preds(model, test_df, test_masks, prep_info)
    
    plot_roc_curve(y_test, y_prob, run_dir / "roc_curve.png")
    plot_pr_curve(y_test, y_prob, run_dir / "pr_curve.png")
    plot_calibration_curve_func(y_test, y_prob, run_dir / "calibration.png", config["model_type"])
    
    # Calibration
    if config.get("calibrate", False):
        # ... existing ...
        pass
        
    # Optional: Mask-Conditioned Conformal Prediction + Risk-Coverage
    from pd_fusion.evaluation.evaluate import compute_risk_coverage
    from pd_fusion.evaluation.plots import plot_risk_coverage
    if config.get("conformal", False):
        from pd_fusion.models.conformal import MaskConformalWrapper
        cp_model = MaskConformalWrapper(model, alpha=0.1)

        # Prepare validation inputs
        val_inputs = None
        if isinstance(prep_info, dict): # MoE
            val_X_d = {}
            mods_used = list(prep_info.keys())
            for mod in mods_used:
                imp, scl, fs = prep_info[mod]
                x, _, _ = preprocess_features(val_df, fs, imp, scl)
                val_X_d[mod] = torch.FloatTensor(x)
            val_inputs = val_X_d
        else:
            imp, scl, fs = prep_info
            x, _, _ = preprocess_features(val_df, fs, imp, scl)
            val_inputs = x
        y_val = val_df[TARGET_COL].values
        try:
            cp_model.fit(val_inputs, y_val, val_masks)
            cp_model.save(run_dir / "conformal_model.pkl")
        except Exception as e:
            logger.warning(f"Conformal calibration skipped due to error: {e}")

    # Risk-Coverage on Test (base model)
    test_inputs = None
    if isinstance(prep_info, dict): # MoE
        test_X_d = {}
        mods_used = list(prep_info.keys())
        for mod in mods_used:
            imp, scl, fs = prep_info[mod]
            x, _, _ = preprocess_features(test_df, fs, imp, scl)
            test_X_d[mod] = torch.FloatTensor(x)
        test_inputs = test_X_d
    else:
        imp, scl, fs = prep_info
        x, _, _ = preprocess_features(test_df, fs, imp, scl)
        test_inputs = x

    if config.get("risk_coverage", True):
        if isinstance(prep_info, dict):
            mods_used = list(prep_info.keys())
            mask_tensor = torch.FloatTensor(np.stack([test_masks[m] for m in mods_used], axis=1))
            # Apply masks to MoE inputs
            masked_test_inputs = {}
            for mod in mods_used:
                x = test_inputs[mod]
                if mod in test_masks:
                    x = x * torch.FloatTensor(test_masks[mod]).unsqueeze(1)
                masked_test_inputs[mod] = x
            y_prob_test = model.predict_proba(masked_test_inputs, mask_tensor)
        elif hasattr(model, "mask_dim"):
            masked = apply_masks_to_matrix(test_inputs, test_masks, prep_info[2])
            y_prob_test = model.predict_proba(masked, masks=get_modality_mask_matrix(test_masks))
        else:
            masked = apply_masks_to_matrix(test_inputs, test_masks, prep_info[2])
            y_prob_test = model.predict_proba(masked, masks=test_masks)
        rc_metrics = compute_risk_coverage(test_df[TARGET_COL].values, y_prob_test, test_masks)
        plot_risk_coverage(rc_metrics, run_dir / "risk_coverage.png")
    
    logger.info(f"Experiment finished. Results saved in {run_dir}")

def _save_run_provenance(run_dir: Path, config: dict, eval_config: dict, dataset_name: str, synthetic: bool, overrides: dict):
    def _git_commit():
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR).decode().strip()
        except Exception:
            return "unknown"

    def _env_info():
        try:
            import torch
            torch_ver = torch.__version__
            cuda_ver = torch.version.cuda
        except Exception:
            torch_ver = "unknown"
            cuda_ver = "unknown"
        return {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch": torch_ver,
            "cuda": cuda_ver,
        }

    command = os.environ.get("PD_FUSION_COMMAND", "unknown")
    provenance = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "command": command,
        "seed": config.get("seed", None),
        "dataset": dataset_name,
        "synthetic": synthetic,
        "overrides": overrides or {},
        "scenarios": eval_config.get("scenarios", []),
        "env": _env_info(),
    }
    save_yaml(config, run_dir / "resolved_config.yaml")
    save_yaml(eval_config, run_dir / "eval_config.yaml")
    save_yaml(provenance, run_dir / "provenance.yaml")

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
    elif dataset_name == "openneuro_ds001907":
        from pd_fusion.data.openneuro_ds001907 import load_openneuro_ds001907
        df, masks = load_openneuro_ds001907(data_config)
    elif dataset_name.startswith("openneuro_") or dataset_name in ["ds004471", "ds004392", "ds001907"]:
        from pd_fusion.data.dev_datasets.openneuro import load_openneuro_dataset
        accession = dataset_name.replace("openneuro_", "")
        df, masks = load_openneuro_dataset(accession)
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
    eval_config_path = config.get("eval_config", "configs/eval_missingness.yaml")
    eval_config = load_yaml(Path(eval_config_path))
    _save_run_provenance(
        run_dir=run_dir,
        config=config,
        eval_config=eval_config,
        dataset_name=dataset_name,
        synthetic=synthetic,
        overrides=overrides,
    )

    # 2. CV Loop
    from pd_fusion.data.splits import get_kfold_splits, get_group_kfold_splits, get_subset_masks
    from pd_fusion.data.schema import TARGET_COL
    metrics_all = []
    
    group_col = config.get("group_col") or config.get("cv_group_col")
    if group_col:
        folds = get_group_kfold_splits(df, n_splits=k, seed=config.get("seed", 42), group_col=group_col)
    else:
        folds = get_kfold_splits(df, n_splits=k, seed=config.get("seed", 42))
    
    # Track fold assignments (validation fold for each sample)
    fold_assign = pd.Series([-1] * len(df), index=df.index, name="fold")
    # Materialize folds to allow assignment and iteration
    folds = list(folds)
    for i, (_, val_df_tmp) in enumerate(folds):
        fold_assign.loc[val_df_tmp.index] = i + 1
    fold_df = df.copy()
    fold_df["fold"] = fold_assign.values
    if group_col and group_col in fold_df.columns:
        fold_df = fold_df[[group_col, "fold", TARGET_COL] + [c for c in ["session"] if c in fold_df.columns]]
    fold_df.to_csv(run_dir / "fold_assignments.csv", index=False)

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

        # Save fold predictions for full_observation
        try:
            from pd_fusion.evaluation.evaluate import predict_proba_for_scenario
            scenario = {"name": "full_observation", "drop_modalities": []}
            y_true, y_prob = predict_proba_for_scenario(model, val_df, val_masks, prep_info, scenario)
            pred_df = pd.DataFrame({
                "y_true": y_true,
                "y_prob": y_prob,
                "fold": i + 1,
            })
            if group_col and group_col in val_df.columns:
                pred_df[group_col] = val_df[group_col].values
            if "session" in val_df.columns:
                pred_df["session"] = val_df["session"].values
            pred_df.to_csv(run_dir / f"preds_fold_{i+1}_full_observation.csv", index=False)
        except Exception as e:
            logger.warning(f"Failed to save fold predictions: {e}")
        
        # Optional: plot example curves for fold 1
    if config.get("cv_plot_example", False) and i == 0:
        plot_degradation_curve({k: v for k, v in results.items() if k != "fold"}, run_dir / "degradation_fold1.png")
        from pd_fusion.evaluation.plots import plot_roc_curve, plot_pr_curve, plot_calibration_curve_func, plot_risk_coverage
        from pd_fusion.evaluation.evaluate import compute_risk_coverage, preprocess_features
        from pd_fusion.data.schema import TARGET_COL, MODALITIES
        import torch
        # Helper to get predictions
        def _get_preds(m, df, ms, p_info):
            is_moe = isinstance(p_info, dict)
            if is_moe:
                X_d = {}
                mods_used = list(p_info.keys())
                for mod in mods_used:
                    imp, scl, fs = p_info[mod]
                    x, _, _ = preprocess_features(df, fs, imp, scl)
                    if mod in ms:
                        x = x * ms[mod].reshape(-1, 1)
                    X_d[mod] = torch.FloatTensor(x)
                m_t = torch.FloatTensor(np.stack([ms[mu] for mu in mods_used], axis=1))
                return m.predict_proba(X_d, m_t)
            else:
                imp, scl, fs = p_info
                x, _, _ = preprocess_features(df, fs, imp, scl)
                x = apply_masks_to_matrix(x, ms, fs)
                if hasattr(m, "mask_dim"):
                    mask_mat = get_modality_mask_matrix(ms)
                    return m.predict_proba(x, mask_mat)
                return m.predict_proba(x, ms)
            
            y_true = val_df[TARGET_COL].values
            y_prob = _get_preds(model, val_df, val_masks, prep_info)
            plot_roc_curve(y_true, y_prob, run_dir / "roc_curve_fold1.png")
            plot_pr_curve(y_true, y_prob, run_dir / "pr_curve_fold1.png")
            plot_calibration_curve_func(y_true, y_prob, run_dir / "calibration_fold1.png", config["model_type"])
            if config.get("risk_coverage", True):
                rc_metrics = compute_risk_coverage(y_true, y_prob, val_masks)
                plot_risk_coverage(rc_metrics, run_dir / "risk_coverage_fold1.png")
        
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
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary_table.csv", index=False)
    summary_df.to_latex(run_dir / "summary_table.tex", index=False, float_format="%.4f")
    
    logger.info(f"CV Finished. Summary saved to {run_dir}")

    # Optional session shift evaluation
    if config.get("session_shift", False):
        session_col = config.get("session_col", "session")
        if session_col in df.columns:
            logger.info("Running session-shift evaluation...")
            for train_ses, test_ses in [(1, 2), (2, 1)]:
                train_df = df[df[session_col] == train_ses]
                val_df = df[df[session_col] == test_ses]
                train_masks = get_subset_masks(masks, train_df.index)
                val_masks = get_subset_masks(masks, val_df.index)
                model, prep_info = train_pipeline(config, train_df, val_df, train_masks, val_masks)
                results = evaluate_model(model, val_df, val_masks, prep_info, eval_config)
                save_yaml(results, run_dir / f"session_shift_ses{train_ses}_to_{test_ses}.yaml")
        else:
            logger.warning(f"session_shift requested but session_col '{session_col}' not found.")
    return aggregated
