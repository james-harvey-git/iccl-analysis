#!/bin/bash
# Eight paired trials, with at most six GPUs allocated to the study at once.
# Required environment: ICCL_PERF_ROOT, ICCL_PERF_CHECKPOINT, ICCL_PERF_EVAL_DIR.
# ICCL_PERF_CANDIDATE selects the candidate that passed the numerical preflight.
#SBATCH --job-name=iccl-perf-train
#SBATCH --array=0-7%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
: "${ICCL_PERF_ROOT:?Set the isolated experiment output directory}"
: "${ICCL_PERF_CHECKPOINT:?Set the full-state continuation checkpoint}"
: "${ICCL_PERF_EVAL_DIR:?Set the validated frozen evaluation directory}"
: "${ICCL_PERF_CANDIDATE:?Select a numerically validated candidate}"
case "${ICCL_PERF_CANDIDATE}" in
    optimized|group2|group4|bias|all_types|bias_group2|all_types_group2) ;;
    *) echo "Unknown candidate: ${ICCL_PERF_CANDIDATE}" >&2; exit 1 ;;
esac

trial="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
variant=baseline
if (( trial % 2 == 1 )); then variant="${ICCL_PERF_CANDIDATE}"; fi
initialization_seed=$((trial / 2))
phase=fresh
target=2000
checkpoint=null
if (( trial >= 6 )); then
    phase=resumed
    initialization_seed=0
    target=3501000
    checkpoint="${ICCL_PERF_CHECKPOINT}"
fi
label="${phase}-seed${initialization_seed}-${variant}"
destination="${ICCL_PERF_ROOT}/trajectories/${label}"
if [[ -e "${destination}/history.jsonl" || -e "${destination}/trajectory.json" ]]; then
    echo "Refusing to overwrite an existing trajectory: ${destination}" >&2
    exit 1
fi
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_RUN_GROUP="${ICCL_PERF_GROUP:-training-performance-parity}"
exec bash scripts/cluster/isambard/run.sh scripts/benchmark_training.py \
    performance.mode=trajectory "performance.variant=${variant}" \
    "performance.initialization_seed=${initialization_seed}" seed=0 \
    "training.num_steps=${target}" "training.resume=${checkpoint}" \
    "data.eval_sets.out_dir=${ICCL_PERF_EVAL_DIR}" \
    "wandb.name=performance-${label}" "hydra.run.dir=${destination}" "$@"
