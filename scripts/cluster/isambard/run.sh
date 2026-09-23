#!/bin/bash
# Shared runtime setup for Isambard jobs, called from the repository root.
# Keep module loads before the host-compiler exports.
set -euo pipefail

iccl_entrypoint="${1:?Usage: run.sh scripts/<entrypoint>.py [Hydra overrides]}"
shift

if ! command -v module >/dev/null 2>&1; then
    if [[ ! -r /opt/cray/pe/lmod/lmod/init/profile ]]; then
        echo "Cannot initialize Isambard's module environment." >&2
        exit 1
    fi
    source /opt/cray/pe/lmod/lmod/init/profile
fi
module load cudatoolkit

# Use GCC 12 for Triton's C launcher and TileLang's NVCC host compiler.
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
"$CC" --version
"$CXX" --version
nvcc --version

export PATH="${HOME}/.local/bin:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
echo "Job ${SLURM_JOB_ID:-local}; node $(hostname); commit $(git rev-parse --short HEAD)"
echo "Entrypoint: ${iccl_entrypoint}; CC=${CC}; CXX=${CXX}"
nvidia-smi

exec uv run --locked python -u "${iccl_entrypoint}" "wandb.mode=${WANDB_MODE:-online}" "$@"
