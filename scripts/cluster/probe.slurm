#!/bin/bash
# Usage: sbatch scripts/cluster/probe.slurm capture|train|eval|benchmark [Hydra overrides]
# Create logs/slurm before submitting. Resource settings can be overridden with sbatch.
# The eight solver threads, two loader workers and one PyTorch thread fit twelve CPUs.
#SBATCH --job-name=iccl-probe
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

case "${1:-}" in
    capture) probe_script="capture_probe_dataset.py" ;;
    train) probe_script="train_probe.py" ;;
    eval) probe_script="eval_probe.py" ;;
    benchmark) probe_script="benchmark_probe.py" ;;
    *) echo "Usage: sbatch scripts/cluster/probe.slurm capture|train|eval|benchmark [overrides]" >&2; exit 2 ;;
esac
shift

probe_repo="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${probe_repo}"
export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
echo "Job ${SLURM_JOB_ID:-local}; commit $(git rev-parse --short HEAD); script ${probe_script}"
nvidia-smi
uv run python "scripts/${probe_script}" "wandb.mode=${WANDB_MODE:-online}" "$@"
