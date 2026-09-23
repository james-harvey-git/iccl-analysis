#!/bin/bash
# Submit from the repository root; create logs/slurm before sbatch.
# Usage: sbatch scripts/cluster/isambard/train_probe.slurm probe.dataset.path=... [overrides]
#SBATCH --job-name=iccl-probe-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
exec bash scripts/cluster/isambard/run.sh scripts/train_probe.py "$@"
