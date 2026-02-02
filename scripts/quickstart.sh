#!/bin/bash
set -e

echo "Running Parkinson's Multimodal Fusion Quickstart (Synthetic)..."

# Ensure src is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the full pipeline with synthetic data
python3 -m pd_fusion.cli run --config configs/quickstart.yaml --synthetic

echo "Quickstart complete. Check the 'runs/' directory for outputs."
