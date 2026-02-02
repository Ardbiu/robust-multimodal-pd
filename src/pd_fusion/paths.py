from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src" / "pd_fusion"
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEV_DATA_DIR = Path(os.environ.get("PD_FUSION_DEV_DATA_DIR", DATA_DIR / "raw_dev"))
RUNS_DIR = ROOT_DIR / "runs"
CONFIGS_DIR = ROOT_DIR / "configs"

def get_run_dir(run_id: str) -> Path:
    run_path = RUNS_DIR / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path
