import subprocess
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("pd_fusion.download")

OPENNEURO_DATASETS = {
    "ds004471": "ds004471", # Parkinson's Disease Multi-Modal
    "ds004392": "ds004392",
    "ds001907": "ds001907"
}

def check_openneuro_cli():
    """Checks if openneuro CLI is installed."""
    if shutil.which("openneuro") is None:
        logger.warning(
            "\n[WARNING] 'openneuro' CLI not found.\n"
            "To download OpenNeuro datasets automatically, please install:\n"
            "  npm install -g @openneuro/cli\n"
            "  openneuro login\n"
            "\nAlternatively, you can skip these datasets or download manually."
        )
        return False
    return True

def download_dataset(accession: str, dest_dir: Path, metadata_only: bool = False):
    if (dest_dir / accession).exists():
         logger.info(f"Dataset {accession} seems to exist at {dest_dir / accession}. Skipping.")
         return

    logger.info(f"Downloading OpenNeuro dataset {accession} to {dest_dir}...")
    try:
        # openneuro download <accession> <output_dir>
        # Note: openneuro cli usually creates the dir.
        cmd = ["openneuro", "download", accession, str(dest_dir / accession)]
        if metadata_only:
            # Attempt to download metadata/tabular files only
            cmd += ["--include", "participants.tsv", "--include", "participants.json", "--include", "dataset_description.json"]
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully downloaded {accession}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {accession}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error downloading {accession}: {e}")

def download_openneuro_datasets(base_dir: Path, metadata_only: bool = False):
    if not check_openneuro_cli():
        return

    neuro_dir = base_dir / "openneuro"
    neuro_dir.mkdir(parents=True, exist_ok=True)

    for name, acc in OPENNEURO_DATASETS.items():
        download_dataset(acc, neuro_dir, metadata_only=metadata_only)
