import argparse
import datetime
import subprocess
from pathlib import Path

MODELS = [
    "unimodal_clinical",
    "unimodal_datspect",
    "unimodal_mri",
    "fusion_late",
    "fusion_masked",
    "fusion_moddrop",
    "moe",
]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}

source ~/.bashrc
conda activate {conda_env} || source activate {conda_env}

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
{export_dev_dir}

set -e
echo "Starting job {job_name}"

{commands}

echo "Job finished"
"""

def build_command(base_config, dataset, synthetic, k_fold, model, seed, output_dir):
    parts = [
        "python -m pd_fusion.cli run",
        f"--config {base_config}",
    ]
    if synthetic:
        parts.append("--synthetic")
    if dataset:
        parts.append(f"--dataset {dataset}")
    if k_fold:
        parts.append(f"--k-fold {k_fold}")
    parts.extend([
        f"--model {model}",
        f"--seed {seed}",
        f"--output-dir {output_dir}",
    ])
    return " \\\n+    ".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Submit two H200 jobs with sequential model runs")
    parser.add_argument("--partition", type=str, default="mit_normal_gpu")
    parser.add_argument("--time", type=str, default="05:00:00")
    parser.add_argument("--mem", type=str, default="64G")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--conda-env", type=str, default="pd_fusion")
    parser.add_argument("--base-config", type=str, default="configs/dev_benchmark_suite.yaml")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--k-fold", type=int, default=None)
    parser.add_argument("--dev-data-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path("runs") / f"dual_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    scripts_dir = sweep_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Build run list and split into 2 roughly equal chunks
    run_list = []
    seeds = [42, 43, 44]
    for model in MODELS:
        for seed in seeds:
            run_list.append((model, seed))

    midpoint = (len(run_list) + 1) // 2
    chunks = [run_list[:midpoint], run_list[midpoint:]]

    for idx, chunk in enumerate(chunks, start=1):
        job_name = f"dual_h200_{idx}"
        commands = []
        for model, seed in chunk:
            output_dir = f"dual_sweep_{timestamp}/{model}_s{seed}"
            commands.append(build_command(
                base_config=args.base_config,
                dataset=args.dataset,
                synthetic=args.synthetic,
                k_fold=args.k_fold,
                model=model,
                seed=seed,
                output_dir=output_dir
            ))
            commands.append("")  # spacer

        export_dev_dir = f"export PD_FUSION_DEV_DATA_DIR={args.dev_data_dir}" if args.dev_data_dir else ""

        script_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            log_dir=logs_dir.absolute(),
            partition=args.partition,
            time_limit=args.time,
            mem=args.mem,
            cpus=args.cpus,
            conda_env=args.conda_env,
            export_dev_dir=export_dev_dir,
            commands="\n".join(commands).strip(),
        )

        script_path = scripts_dir / f"{job_name}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        if args.dry_run:
            print(f"[DRY RUN] Generated {script_path}")
        else:
            print(f"Submitting {job_name}...")
            subprocess.run(["sbatch", str(script_path)], check=False)

    print(f"Results will be in {sweep_dir}")

if __name__ == "__main__":
    main()
