import argparse
import logging
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


    # Dev Datasets
    download_parser = subparsers.add_parser("download-dev")
    download_parser.add_argument("--dataset", type=str, default="all")
    download_parser.add_argument("--out", type=str, default="data/raw_dev")
    
    prepare_parser = subparsers.add_parser("prepare-dev")
    
    args = parser.parse_args()
    setup_logging()
    
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
            download_openneuro_datasets(out_dir)
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
        if args.model: overrides["model_type"] = args.model
        if args.seed is not None: overrides["seed"] = args.seed
        if args.output_dir: overrides["output_dir"] = args.output_dir
        
        run_full_pipeline(args.config, args.synthetic, overrides=overrides)
    else:
        if args.command is None:
            parser.print_help()
        else:
            print("Command not implemented yet.")

if __name__ == "__main__":
    main()
