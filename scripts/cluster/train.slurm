#!/bin/bash
# SLURM job for ICCL meta-training on the EIT CDT cluster (compute partition,
# nodes with 8x H100).
#
# Usage:
#   sbatch scripts/cluster/train.slurm [hydra-overrides...]
#
# Examples:
#   # Pilot training run with the defaults from configs/
#   sbatch scripts/cluster/train.slurm
#
#   # Short smoke run to validate the pipeline + W&B logging end to end
#   sbatch scripts/cluster/train.slurm training.num_steps=2000 \
#       training.validation_every=500 training.monitor_every=1000 \
#       training.checkpoint_every=1000
#
#   # Resume a run from its checkpoint
#   sbatch scripts/cluster/train.slurm \
#       training.resume=outputs/<date>/<time>/checkpoints/last.pt
#
#   # Name the run, and keep its final weights as a W&B artifact
#   sbatch scripts/cluster/train.slurm wandb.name=pilot-logsnapshots \
#       wandb.upload_weights=true
#
#   # Log-spaced snapshots, for watching a capability emerge early in training
#   sbatch scripts/cluster/train.slurm \
#       'training.snapshots.log={count: 10, start: 100}'
#
#   # W&B is online by default for cluster runs; switch via the env var
#   # (an override in "$@" would collide with the one this script passes)
#   WANDB_MODE=disabled sbatch scripts/cluster/train.slurm
#
# Isambard (8x H100 -> H200): machine-specific values live only in the #SBATCH
# directives below — override at submit time (sbatch --partition=... --time=...)
# or add a sibling .slurm once its partition/account names are known.
#
# ── One-time setup (once per cluster) ─────────────────────────────────────────
#
#   # 1. Install uv
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
#   # 2. Clone and install dependencies
#   cd $HOME && git clone git@github.com:james-harvey-git/iccl-analysis.git
#   cd iccl-analysis && uv sync
#
#   # 3. Log in to W&B
#   uv run wandb login
#
#   # 4. Create the SLURM log directory (must exist before the first job starts)
#   mkdir -p logs/slurm
#
# ── Job metadata ──────────────────────────────────────────────────────────────
#SBATCH --job-name=iccl-train
#SBATCH --partition=compute
#SBATCH --time=12:00:00

# ── Resources ─────────────────────────────────────────────────────────────────
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8       # on-the-fly CPU sequence generation (dataloader workers + main)
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ── Logging ───────────────────────────────────────────────────────────────────
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${HOME}/iccl-analysis"
cd "${REPO_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "============================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU(s):   ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "Commit:   $(git rev-parse --short HEAD)"
echo "Dir:      $(pwd)"
nvidia-smi
echo "============================================"

# ── Frozen eval sets ──────────────────────────────────────────────────────────
# Prepare once with scripts/make_eval_sets.py and the same data/seed overrides.
# Training verifies the configured bundle's manifest before its first step;
# stale or missing data fails explicitly rather than triggering a rebuild.

# ── Train ─────────────────────────────────────────────────────────────────────
uv run python scripts/train.py "wandb.mode=${WANDB_MODE:-online}" "$@"
