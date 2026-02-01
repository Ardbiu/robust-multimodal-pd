#!/bin/bash
set -e

MODELS=("configs/model_unimodal.yaml" "configs/model_fusion.yaml" "configs/model_moe.yaml")

for cfg in "${MODELS[@]}"; do
    echo "Running model config: $cfg"
    python -m pd_fusion.cli run --config "$cfg" --synthetic
done

echo "All experiments complete."
活跃的
