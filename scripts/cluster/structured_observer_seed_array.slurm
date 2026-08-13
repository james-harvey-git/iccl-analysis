#!/bin/bash
# SLURM array for one independently resumable structured-observer SMC seed per GPU.
#
# Usage:
#   sbatch scripts/cluster/structured_observer_seed_array.slurm
#
# Array tasks generate separate content-addressed caches for seeds 0, 1, and 2.
# After all tasks finish, pass the three reported cache paths to:
#   uv run python scripts/merge_structured_observer_seeds.py <seed-cache>...

#SBATCH --job-name=iccl-gp-seed
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${HOME}/iccl-analysis"
cd "${REPO_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1

SMC_SEED="${SLURM_ARRAY_TASK_ID}"

echo "Job:      ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "SMC seed: ${SMC_SEED}"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "Dir:      $(pwd)"
nvidia-smi

if [ ! -f data/eval_sets/in_dist.npz ] || [ ! -f data/eval_sets/in_dist.meta.json ]; then
    echo "Missing frozen in_dist suite; generating frozen evaluation sets."
    uv run python scripts/make_eval_sets.py
fi

uv run python scripts/make_structured_observer.py \
    structured_observer.device=cuda \
    "structured_observer.smc.seeds=[${SMC_SEED}]" \
    "$@"
