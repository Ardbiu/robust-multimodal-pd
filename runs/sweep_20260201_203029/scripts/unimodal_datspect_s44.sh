#!/bin/bash
#SBATCH --job-name=unimodal_datspect_s44
#SBATCH --output=/Users/aradhyadixit/Developer/IEEE-spid/runs/sweep_20260201_203029/logs/unimodal_datspect_s44.out
#SBATCH --error=/Users/aradhyadixit/Developer/IEEE-spid/runs/sweep_20260201_203029/logs/unimodal_datspect_s44.err
#SBATCH --partition=sched_mit_hill
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate pd_fusion || source activate pd_fusion

echo "Starting job unimodal_datspect_s44"
echo "Model: unimodal_datspect"
echo "Seed: 44"

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python -m pd_fusion.cli run \
    --config configs/model_fusion.yaml \
    --synthetic \
    --model unimodal_datspect \
    --seed 44 \
    --output-dir runs/sweep_20260201_203029/unimodal_datspect_s44

echo "Job finished"
