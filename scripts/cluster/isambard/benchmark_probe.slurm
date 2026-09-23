#!/bin/bash
# Submit from the repository root; create logs/slurm before sbatch.
# Usage: sbatch scripts/cluster/isambard/benchmark_probe.slurm [Hydra overrides]
#SBATCH --job-name=iccl-probe-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
exec bash scripts/cluster/isambard/run.sh scripts/benchmark_probe.py "$@"
