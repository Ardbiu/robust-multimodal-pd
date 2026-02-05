#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/ppmi_studydata.yaml}

python scripts/ppmi_build_dataset.py --config "$CONFIG" --seed 42
python scripts/ppmi_train_tabular.py --config "$CONFIG" --seed 42 --limit 200
python scripts/ppmi_eval_report.py --config "$CONFIG" --out_dir $(ls -td runs/ppmi_tabular_* | head -1)
