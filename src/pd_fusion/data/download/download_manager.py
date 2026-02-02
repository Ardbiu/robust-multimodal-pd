import logging
import argparse
from pathlib import Path
from pd_fusion.data.download.uci_download import download_uci_datasets
from pd_fusion.data.download.openneuro_download import download_openneuro_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pd_fusion.download_manager")

def print_manual_instructions():
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED FOR RESTRICTED DATASETS")
    print("="*60)
    print("1. Synapse mPower (Mobile Parkinson's Data)")
    print("   - URL: https://www.synapse.org/#!Synapse:syn4993293")
    print("   - Requires: Synapse account, Certified User status, Accepted Conditions.")
    print("   - Instruction: Download tables/images and place in 'data/raw_dev/synapse/'")
    print("\n2. BioFIND (LONI/IDA)")
    print("   - URL: https://ida.loni.usc.edu/")
    print("   - Requires: Data Use Agreement (DUA).")
    print("   - Instruction: Download archive and place in 'data/raw_dev/biofind/'")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Download Development Datasets")
    parser.add_argument("--out", type=str, default="data/raw_dev", help="Output directory")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "uci", "openneuro", "manual"])
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading datasets to {out_dir}...")
    
    if args.dataset in ["all", "uci"]:
        logger.info("--- Processing UCI Datasets ---")
        download_uci_datasets(out_dir)
        
    if args.dataset in ["all", "openneuro"]:
        logger.info("--- Processing OpenNeuro Datasets ---")
        download_openneuro_datasets(out_dir)
        
    if args.dataset in ["all", "manual"]:
        print_manual_instructions()

if __name__ == "__main__":
    main()
