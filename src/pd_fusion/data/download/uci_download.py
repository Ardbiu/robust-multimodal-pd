import os
import requests
import logging
from pathlib import Path

logger = logging.getLogger("pd_fusion.download")

UCI_URLS = {
    "parkinsons": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
    "telemonitoring": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
}

def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            os.remove(dest_path) # Cleanup partial
        raise

def download_uci_datasets(base_dir: Path):
    """
    Downloads UCI Parkinsons datasets to base_dir/uci/
    """
    uci_dir = base_dir / "uci"
    uci_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parkinsons (Voice -> PD/Control)
    download_file(UCI_URLS["parkinsons"], uci_dir / "parkinsons.data")

    # 2. Telemonitoring (Voice -> UPDRS)
    download_file(UCI_URLS["telemonitoring"], uci_dir / "parkinsons_updrs.data")
