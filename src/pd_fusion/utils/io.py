import yaml
import json
import pickle
from pathlib import Path
from typing import Any, Dict

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: Path):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

def save_pickle(obj: Any, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
