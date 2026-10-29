#!/bin/bash
# Submit from the repository root with a prepared environment and evaluation bundle.
# Usage: WANDB_MODE=disabled sbatch scripts/cluster/isambard/performance.slurm [overrides]
#SBATCH --job-name=iccl-performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec bash scripts/cluster/isambard/run.sh scripts/benchmark_training.py "$@"
