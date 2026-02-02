#!/bin/bash
#SBATCH --job-name=unimodal_mri_s43
#SBATCH --output=/Users/aradhyadixit/Developer/IEEE-spid/runs/sweep_20260201_203029/logs/unimodal_mri_s43.out
#SBATCH --error=/Users/aradhyadixit/Developer/IEEE-spid/runs/sweep_20260201_203029/logs/unimodal_mri_s43.err
#SBATCH --partition=sched_mit_hill
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate pd_fusion || source activate pd_fusion

echo "Starting job unimodal_mri_s43"
echo "Model: unimodal_mri"
echo "Seed: 43"

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python -m pd_fusion.cli run \
    --config configs/model_fusion.yaml \
    --synthetic \
    --model unimodal_mri \
    --seed 43 \
    --output-dir runs/sweep_20260201_203029/unimodal_mri_s43

echo "Job finished"
