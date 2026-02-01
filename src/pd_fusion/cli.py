import argparse
import logging
from pd_fusion.utils.logging import setup_logging
from pd_fusion.utils.io import load_yaml
from pd_fusion.paths import CONFIGS_DIR
from pd_fusion.experiments.run_experiment import run_full_pipeline

def main():
    parser = argparse.ArgumentParser(description="PPMI Multimodal Fusion CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Prepare data
    prep_parser = subparsers.add_parser("prepare-data")
    prep_parser.add_argument("--config", type=str, required=True)
    prep_parser.add_argument("--synthetic", action="store_true")

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

    args = parser.parse_args()
    setup_logging()
    
    if args.command == "run":
        run_full_pipeline(args.config, args.synthetic)
    else:
        print("Please use 'run' command for the skeleton demonstration.")
        # Other commands would call specific parts of run_full_pipeline

if __name__ == "__main__":
    main()
活跃的
