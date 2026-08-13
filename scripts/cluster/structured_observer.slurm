#!/bin/bash
# SLURM job for the full structured-observer cache on one CUDA GPU.
#
# Usage:
#   sbatch scripts/cluster/structured_observer.slurm [hydra-overrides...]
#
# Default: 256 in-distribution sequences, 2048 prior-matched random features,
# 1024 particles, and SMC seeds 0/1/2. The job writes only local cache files;
# it creates no W&B run. Copy `outputs/structured_observer/` back before plotting.
#
# Smaller end-to-end smoke:
#   sbatch scripts/cluster/structured_observer.slurm \
#       structured_observer.sequence_limit=2 \
#       structured_observer.kernel.num_features=128 \
#       structured_observer.smc.num_particles=64 \
#       'structured_observer.smc.seeds=[0]'

#SBATCH --job-name=iccl-gp-observer
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${HOME}/iccl-analysis"
cd "${REPO_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1

echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "Dir:      $(pwd)"
nvidia-smi

if [ ! -f data/eval_sets/in_dist.npz ] || [ ! -f data/eval_sets/in_dist.meta.json ]; then
    echo "Missing frozen in_dist suite; generating frozen evaluation sets."
    uv run python scripts/make_eval_sets.py
fi

uv run python scripts/make_structured_observer.py structured_observer.device=cuda "$@"
