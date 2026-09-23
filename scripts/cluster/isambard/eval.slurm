#!/bin/bash
# Submit from the repository root; create logs/slurm before sbatch.
# Usage: sbatch scripts/cluster/isambard/eval.slurm 'evaluation.checkpoints=[...]' [overrides]
#SBATCH --job-name=iccl-gdn-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
exec bash scripts/cluster/isambard/run.sh scripts/eval.py "$@"
