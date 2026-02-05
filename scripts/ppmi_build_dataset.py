import argparse
import logging
from pathlib import Path
import yaml

from pd_fusion.data.ppmi_studydata import build_ppmi_datasets


def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppmi_build")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")

    if not logger.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        logger.addHandler(stream)

        file_handler = logging.FileHandler(out_dir / "ppmi_build_dataset.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PPMI study-data datasets")
    parser.add_argument("--config", required=True, help="Path to ppmi_studydata.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    parser.add_argument("--out_dir", default=None, help="Override processed_ppmi_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.out_dir:
        cfg["processed_ppmi_dir"] = args.out_dir
    if args.seed is not None:
        cfg.setdefault("splits", {})
        cfg["splits"]["seeds"] = [args.seed]

    out_dir = Path(cfg["processed_ppmi_dir"])
    logger = setup_logging(out_dir)

    logger.info("Building PPMI datasets with config: %s", args.config)
    outputs = build_ppmi_datasets(cfg, logger)
    for key, path in outputs.items():
        logger.info("Saved %s -> %s", key, path)


if __name__ == "__main__":
    main()
