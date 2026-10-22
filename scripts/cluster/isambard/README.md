# Isambard GPU jobs

Submit these scripts from the repository root. Each launcher defaults to one GPU on one
node and uses the site's default partition and account.
The scripts explicitly request host memory per node and CPU counts for their
worker/thread budgets. See
[Isambard's batch-job guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm/#running-a-single-batch-job).

| Job | Script | CPUs per task | Host RAM | Walltime |
| --- | --- | ---: | ---: | ---: |
| GDN training | `train.slurm` | 8 | 64 GiB | 12 hours |
| GDN evaluation | `eval.slurm` | 4 | 64 GiB | 4 hours |
| Probe dataset capture | `capture_probe.slurm` | 4 | 32 GiB | 15 minutes |
| Probe training | `train_probe.slurm` | 12 | 64 GiB | 12 hours |
| Probe evaluation | `eval_probe.slurm` | 12 | 32 GiB | 4 hours |
| Probe benchmark | `benchmark_probe.slurm` | 12 | 64 GiB | 1 hour |

Each `#SBATCH --mem=...` sets total host RAM per node. GPU memory is a separate
resource. These are conservative starting budgets for the reference workloads,
with headroom for loader workers, mapped dataset shards and checkpoint loading
or serialization. Explicit requests avoid inheriting an oversized site default.
Override them with `sbatch --mem=...` before the script name when a workload
needs a different budget.

The capture default is a bounded starting allocation, not a measured completion
estimate. Set walltime from the pilot before requesting a larger dataset; cold
kernel compilation can dominate a short capture.

Slurm records resource requests when a job is submitted; editing a script does
not change queued jobs. For a pending job, set its per-node memory request in
MiB with `scontrol update JobId=<job-id> MinMemoryNode=32768` for 32 GiB, or
`MinMemoryNode=65536` for 64 GiB, then verify `ReqTRES` with
`scontrol show job <job-id>`. This preserves its job ID and dependencies. See
the [scontrol documentation](https://slurm.schedmd.com/scontrol.html#OPT_MinMemoryNode).

All launchers delegate to `run.sh`, which initializes the module command if
needed, loads `cudatoolkit`, then exports `CC=/usr/bin/gcc-12` and
`CXX=/usr/bin/g++-12`. It checks the compiler commands before starting Python,
records their versions, the Git commit and GPU information in the job log, and
uses `uv run --locked --no-sync`. Add any additional module loads before those compiler
exports; future Isambard launchers should also delegate to this runner.

Jobs use the prepared Python environment without syncing dependencies. Concurrent
jobs share the checkout's `.venv`, and an automatic reinstall during another job's
PyTorch imports can temporarily remove a required CUDA library. `--locked` prevents
lockfile updates but still permits environment changes; `--no-sync` prevents those
changes at launch. See [uv's locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

Run `uv sync --locked` once before submission and after dependency changes, **only
while no jobs are using that environment**. While jobs are running, use
`uv run --locked --no-sync ...` for other commands in the same checkout as well.
If dependencies must change during a running experiment, prepare a separate
checkout and environment for the new jobs.

W&B defaults to online. Set `WANDB_MODE=offline` or `WANDB_MODE=disabled` when
submitting to select another mode. Model/data/probe settings are ordinary Hydra
arguments after the script name. Scheduler overrides such as `--time`,
`--mem`, `--cpus-per-task`, `--partition` or `--account` go before the script name. Set
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` explicitly if needed; both default to one.

Create the log directory before submission: Slurm opens these files before the
script starts. Logs are `logs/slurm/<job-name>_<job-id>.out` and `.err`.

```bash
mkdir -p logs/slurm
uv sync --locked
```

## GDN training and evaluation

Prepare the frozen evaluation bundle in a compute allocation with
`scripts/make_eval_sets.py` and the same data settings as training. Repeating
preparation verifies/reuses a matching bundle; the job launchers do not generate
it. A default-data run is:

```bash
uv run --locked --no-sync python scripts/make_eval_sets.py
sbatch scripts/cluster/isambard/train.slurm wandb.name=reference-gdn
```

Evaluate a local or `wandb://` GDN checkpoint, keeping the model/data settings
consistent with its training configuration:

```bash
sbatch scripts/cluster/isambard/eval.slurm \
  'evaluation.checkpoints=[/path/to/gdn/checkpoint.pt]' \
  evaluation.suites=all
```

## Fixed-world task-vector probes

The [probe guide](../../../docs/task-vector-probes.md) defines the sampling,
three independent stages, controls, monitoring and world-averaged reports.
The scheduler exposes four GPUs per node. The submission helper requests one
worker per GPU, using a second node for five to eight GPUs. It refuses more than
eight GPUs, idle workers beyond the world count, or another queued/running probe
chain for this account. The initial pilot uses **one GPU**.

Use scratch for data, results, logs and caches. Keep the existing immutable GDN
checkpoint reference. From the repository root, this submits three separate
jobs whose dependencies permit only one stage to run at a time:

```bash
: "${SCRATCHDIR:?Set the Isambard scratch directory}"
task_probe_root="$SCRATCHDIR/iccl-analysis/task-vector/pilot-unique"
task_probe_checkpoint="$PWD/artifacts/weights-variable-surplus-s0-4-400k:v14/step_2900000.pt"
export WANDB_MODE=online
export WANDB_CACHE_DIR="$SCRATCHDIR/iccl-analysis/cache/wandb"
export WANDB_DATA_DIR="$SCRATCHDIR/iccl-analysis/cache/wandb-staging"
export WANDB_ARTIFACT_DIR="$SCRATCHDIR/iccl-analysis/artifacts"
export UV_CACHE_DIR="$SCRATCHDIR/iccl-analysis/cache/uv"
export MPLCONFIGDIR="$SCRATCHDIR/iccl-analysis/cache/matplotlib"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" \
  "$UV_CACHE_DIR" "$MPLCONFIGDIR"

uv run --locked --no-sync python scripts/cluster/isambard/submit_probe.py \
  --root "$task_probe_root" --checkpoint "$task_probe_checkpoint" --preset pilot \
  --gpus 1 --cpus-per-task 4 --memory-per-node 32G \
  --capture-time 00:15:00 --train-time 00:15:00 --eval-time 00:10:00 \
  experiment.name=task-vector-pilot
```

The explicit times are pilot bounds, not throughput estimates. The helper saves
accepted job IDs immediately in `submission.json`; if submission is interrupted,
inspect that file and `squeue` before resubmitting. Failed dependencies are cancelled.
The shared study lock also prevents competing W&B writers. Resume uses the same
study path and scientific settings; rerun the chain only after earlier jobs have
finished or been deliberately cancelled.

For a bounded end-to-end timing run, submit the benchmark as a single job:

```bash
mkdir -p "$SCRATCHDIR/iccl-analysis/logs/slurm"
sbatch --gpus=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:20:00 \
  --output="$SCRATCHDIR/iccl-analysis/logs/slurm/%x_%j.out" \
  --error="$SCRATCHDIR/iccl-analysis/logs/slurm/%x_%j.err" \
  scripts/cluster/isambard/benchmark_probe.slurm probe=pilot \
  "experiment.root=$SCRATCHDIR/iccl-analysis/task-vector/timing-unique" \
  "probe.dataset.checkpoint=$PWD/artifacts/weights-variable-surplus-s0-4-400k:v14/step_2900000.pt" \
  device=cuda
```

This measures capture, fitting, evaluation/rendering and warmed direct/CG solves.
The benchmark requires a fresh path so it cannot mistake a resumed stage for
throughput. Full-dimensional state features are retained. Choose larger
development and confirmation budgets only after inspecting these measurements.

For multiple workers, request enough worlds and use the helper's `--gpus` flag.
The `srun` launchers bind exactly one visible GPU per process and shard by world;
they do not use DDP. Memory requests are total **per node** and must account for
all workers on that node. Shared numerical artifacts and event streams stay on
scratch; one coordinator resumes the same W&B run across stages and phases.

Keep login-node work to editing, Git, lightweight metadata checks and submission.
Run model capture, probe fitting, numerical tests and benchmark/render workloads
inside allocations. Do not update the shared checkout or synchronize its
environment while a job is using it.
