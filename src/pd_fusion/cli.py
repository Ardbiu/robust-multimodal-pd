import argparse
import logging
from pathlib import Path
from pd_fusion.utils.logging import setup_logging
from pd_fusion.utils.io import load_yaml
from pd_fusion.paths import CONFIGS_DIR
from pd_fusion.experiments.run_experiment import run_full_pipeline

def main():
    parser = argparse.ArgumentParser(description="PPMI Multimodal Fusion CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Prepare data (process-and-merge)
    validate_parser = subparsers.add_parser("validate-data")
    validate_parser.add_argument("--config", type=str, required=True, help="Data config (sources)")
    validate_parser.add_argument("--columns", type=str, default="configs/ppmi_columns.yaml", help="Column mapping config")

    # Train
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--data-config", type=str, default="configs/data_ppmi.yaml")
    train_parser.add_argument("--synthetic", action="store_true")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", type=str, required=True)
    eval_parser.add_argument("--run-dir", type=str, required=True)

    # Full Run
    full_parser = subparsers.add_parser("run")
    full_parser.add_argument("--config", type=str, required=True)
    full_parser.add_argument("--synthetic", action="store_true")
    full_parser.add_argument("--model", type=str, help="Override model type")
    full_parser.add_argument("--seed", type=int, help="Override random seed")
    full_parser.add_argument("--output-dir", type=str, help="Override output directory name")
    full_parser.add_argument("--k-fold", type=int, help="Run K-Fold CV (e.g. 5)")
    full_parser.add_argument("--dataset", type=str, help="Override dataset name (e.g., uci_parkinsons, openneuro_ds004471)")


    # Dev Datasets
    download_parser = subparsers.add_parser("download-dev")
    download_parser.add_argument("--dataset", type=str, default="all")
    download_parser.add_argument("--out", type=str, default="data/raw_dev")
    download_parser.add_argument("--openneuro-metadata-only", action="store_true")
    
    prepare_parser = subparsers.add_parser("prepare-dev")
    
    args = parser.parse_args()
    setup_logging()
    # Capture command for provenance
    import os as _os
    import sys as _sys
    _os.environ["PD_FUSION_COMMAND"] = "python -m pd_fusion.cli " + " ".join(_sys.argv[1:])
    
    if args.command == "download-dev":
        from pd_fusion.data.download.download_manager import main as download_main
        # Re-parse args inside logic or call directly?
        # download_manager uses argparse too. Let's call its logic directly if refactored, 
        # or just invoke it.
        # Ideally download_manager.download_all(args.dataset, args.out)
        # For this skeleton, I will execute the script module logic adapting args.
        from pd_fusion.data.download.download_manager import download_uci_datasets, download_openneuro_datasets, print_manual_instructions
        from pathlib import Path
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.dataset in ["all", "uci"]:
            download_uci_datasets(out_dir)
        if args.dataset in ["all", "openneuro"]:
            download_openneuro_datasets(out_dir, metadata_only=args.openneuro_metadata_only)
        if args.dataset in ["all", "manual"]:
            print_manual_instructions()
            
    elif args.command == "validate-data":
        from pd_fusion.data.ppmi_loader import process_and_merge_data
        from pathlib import Path
        data_conf = load_yaml(Path(args.config))
        col_conf = load_yaml(Path(args.columns))
        process_and_merge_data(data_conf, col_conf)
    
    elif args.command == "run":
        # Pass overrides as dict
        overrides = {}
        def _resolve_path(path_str: str) -> Path:
            p = Path(path_str)
            if p.exists():
                return p
            from pd_fusion.paths import ROOT_DIR
            p2 = ROOT_DIR / p
            return p2

        def _get_unimodal_backbone(config_path: str) -> str:
            try:
                cfg = load_yaml(_resolve_path(config_path))
                return str(cfg.get("unimodal_backbone", "gbdt")).lower()
            except Exception:
                return "gbdt"

        if args.model:
            def _load_params(path_str: str):
                try:
                    p = Path(path_str)
                    if not p.exists():
                        from pd_fusion.paths import ROOT_DIR
                        p = ROOT_DIR / p
                    conf = load_yaml(p)
                    return conf.get("params", {})
                except Exception:
                    return {}

            if args.model.startswith("unimodal_") and args.model != "unimodal_gbdt":
                raw_modality = args.model.replace("unimodal_", "")
                backbone = "gbdt"
                if raw_modality.endswith("_mlp"):
                    backbone = "mlp"
                    raw_modality = raw_modality.replace("_mlp", "")
                elif raw_modality.endswith("_gbdt"):
                    backbone = "gbdt"
                    raw_modality = raw_modality.replace("_gbdt", "")
                else:
                    backbone = _get_unimodal_backbone(args.config)
                overrides["modality"] = raw_modality
                if backbone == "mlp":
                    overrides["model_type"] = "unimodal_mlp"
                    overrides["params"] = _load_params("configs/model_fusion.yaml")
                else:
                    overrides["model_type"] = "unimodal_gbdt"
                    overrides["params"] = _load_params("configs/model_unimodal.yaml")
            elif args.model in ["fusion_late", "fusion_masked", "fusion_moddrop"]:
                overrides["model_type"] = args.model
                overrides["params"] = _load_params("configs/model_fusion.yaml")
            elif args.model == "moe":
                overrides["model_type"] = args.model
                overrides["params"] = _load_params("configs/model_moe.yaml")
            else:
                overrides["model_type"] = args.model
        if args.seed is not None: overrides["seed"] = args.seed
        if args.output_dir: overrides["output_dir"] = args.output_dir
        if args.dataset: overrides["dataset"] = args.dataset
        
        # Check config-driven CV if no CLI override
        config_k = None
        if args.k_fold is None:
            try:
                conf = load_yaml(Path(args.config))
                config_k = conf.get("cv_folds") or conf.get("k_folds")
            except Exception:
                config_k = None

        if args.k_fold is not None:
            # Run CV
            from pd_fusion.experiments.run_experiment import run_cv_pipeline
            run_cv_pipeline(args.config, k=args.k_fold, synthetic=args.synthetic, overrides=overrides)
        elif config_k is not None:
            from pd_fusion.experiments.run_experiment import run_cv_pipeline
            run_cv_pipeline(args.config, k=int(config_k), synthetic=args.synthetic, overrides=overrides)
        else:
            # Run Single Split
            run_full_pipeline(args.config, args.synthetic, overrides=overrides)
    else:
        if args.command is None:
            parser.print_help()
        else:
            print("Command not implemented yet.")

if __name__ == "__main__":
    main()
