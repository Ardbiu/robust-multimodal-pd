import os
import subprocess
import argparse
from pathlib import Path
import datetime

# Experiment Grid
SEEDS = [42, 43, 44]
MODELS = [
    "unimodal_clinical", 
    "unimodal_datspect", 
    "unimodal_mri", 
    "fusion_late", 
    "fusion_moddrop", 
    "moe"
]

# SLURM Configuration
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate pd_fusion || source activate pd_fusion

echo "Starting job {job_name}"
echo "Model: {model}"
echo "Seed: {seed}"

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python -m pd_fusion.cli run \\
    --config {config_path} \\
    --synthetic \\
    --model {model} \\
    --seed {seed} \\
    --output-dir {output_dir}

echo "Job finished"
"""

def main():
    parser = argparse.ArgumentParser(description="Submit optimization sweep to SLURM")
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not submit")
    parser.add_argument("--partition", type=str, default="sched_mit_hill", help="SLURM partition")
    parser.add_argument("--base-config", type=str, default="configs/model_fusion.yaml")
    args = parser.parse_args()

    # Create logs directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path("runs") / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    scripts_dir = sweep_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    print(f"Generating sweep in {sweep_dir}")

    for model in MODELS:
        for seed in SEEDS:
            job_name = f"{model}_s{seed}"
            output_dir = f"runs/sweep_{timestamp}/{job_name}"
            
            script_content = SLURM_TEMPLATE.format(
                job_name=job_name,
                log_dir=logs_dir.absolute(),
                partition=args.partition,
                model=model,
                seed=seed,
                config_path=args.base_config,
                output_dir=output_dir
            )
            
            script_path = scripts_dir / f"{job_name}.sh"
            with open(script_path, "w") as f:
                f.write(script_content)
            
            if args.dry_run:
                print(f"[DRY RUN] Generated {script_path}")
            else:
                print(f"Submitting {job_name}...")
                subprocess.run(["sbatch", str(script_path)])

    print(f"\nPro-tip: Monitor jobs with 'squeue -u $USER'")
    print(f"Results will be in {sweep_dir}")

if __name__ == "__main__":
    main()
