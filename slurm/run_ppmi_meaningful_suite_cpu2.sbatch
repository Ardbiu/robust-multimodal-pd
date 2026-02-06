#!/bin/bash
#SBATCH --job-name=ppmi_suite_cpu2
#SBATCH --partition=mit_normal_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/adixit1/IEEE-spid/runs/slurm_logs/ppmi_suite_cpu2_%j.out
#SBATCH --error=/home/adixit1/IEEE-spid/runs/slurm_logs/ppmi_suite_cpu2_%j.err

set -euo pipefail

mkdir -p /home/adixit1/IEEE-spid/runs/slurm_logs

module load deprecated-modules anaconda3/2022.05-x86_64
source /nfs/software001/home/software-r8-x86_64/spack-20230328/opt/spack/linux-rocky8-x86_64/gcc-8.5.0/anaconda3-2022.05-auh4o3tsby7ze6q6v3stn2hhvvnpoy5f/etc/profile.d/conda.sh
conda activate base

export PYTHONPATH=$PYTHONPATH:/home/adixit1/IEEE-spid/src

python /home/adixit1/IEEE-spid/scripts/ppmi_meaningful_suite.py \
  --input-csv /home/adixit1/IEEE-spid/data/processed/ppmi/ppmi_subject_baseline.csv \
  --output-dir /home/adixit1/IEEE-spid/runs/ppmi_meaningful_suite_$(date +%Y%m%d_%H%M%S) \
  --num-threads 2
