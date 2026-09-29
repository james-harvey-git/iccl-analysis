# Isambard GPU jobs

Submit these scripts from the repository root. Each job requests one GPU on one
node, runs one Python process, and uses the site's default partition and account.
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

The 15-minute capture default targets the reference 12,000-episode dataset.
For larger datasets, override `--time` before the script name using timings from
the capture run's `capture.json`; short captures can be dominated by startup.

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

## Probe dataset, training and evaluation

The set decoder uses **M=4, T=7, D=32**, with state capture after the final
boundary and a 1,344-output affine decoder. `module-set-decoder-v1` requires a
fresh M=4 dataset and new training. Existing M=8 states/checkpoints are rejected;
leave previous results intact. The [probe guide](../../../docs/module-decoder.md)
describes the exact matching objective, controls, plots and storage contract.

The following **Bash** block queues a complete pilot with distinct W&B names.
Replace the GDN reference with an immutable artifact version or a local path.
Run it from the repository checkout. Datasets, Hydra outputs, Slurm logs,
artifact downloads and the listed caches go directly to `$SCRATCHDIR`; no home
directory symlink is needed. W&B is explicitly online and full checkpoint/data
uploads stay disabled.

```bash
bash <<'BASH'
set -euo pipefail
: "${SCRATCHDIR:?SCRATCHDIR must point to your Isambard scratch directory}"
set_probe_gdn='wandb://ENTITY/PROJECT/ARTIFACT:VERSION'  # Or /absolute/path/to/gdn.pt.
set_probe_root="$SCRATCHDIR/iccl-analysis"
set_probe_train_count=10000
set_probe_steps=10000
set_probe_shard=512
set_probe_capture_time=00:15:00
set_probe_capture_mem=32G
set_probe_train_time=02:00:00
set_probe_eval_time=01:00:00
set_probe_run="set-m4t7d32-n${set_probe_train_count}-b128-u${set_probe_steps}-s0-$(date +%Y%m%d-%H%M%S)"
set_probe_dataset="$set_probe_root/module-decoder/datasets/$set_probe_run"
set_probe_runs="$set_probe_root/module-decoder/runs/$set_probe_run"
set_probe_logs="$set_probe_root/logs/slurm"

export WANDB_MODE=online
export WANDB_CACHE_DIR="$set_probe_root/cache/wandb"
export WANDB_DATA_DIR="$set_probe_root/cache/wandb-staging"
export WANDB_ARTIFACT_DIR="$set_probe_root/artifacts"
export UV_CACHE_DIR="$set_probe_root/cache/uv"
mkdir -p "$set_probe_logs" "$set_probe_runs" "$WANDB_CACHE_DIR" \
  "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$UV_CACHE_DIR"

set_probe_common=(
  "probe.dataset.path=$set_probe_dataset"
  "probe.solver.cache_dir=$set_probe_root/cache/probe-assignment"
  seed=0
)
set_probe_log_args=(
  "--output=$set_probe_logs/%x_%j.out"
  "--error=$set_probe_logs/%x_%j.err"
)

set_probe_capture_job=$(sbatch --parsable "${set_probe_log_args[@]}" \
  --cpus-per-task=4 "--mem=$set_probe_capture_mem" "--time=$set_probe_capture_time" \
  scripts/cluster/isambard/capture_probe.slurm \
  "${set_probe_common[@]}" "probe.capture.checkpoint=$set_probe_gdn" \
  "probe.dataset.counts.train=$set_probe_train_count" \
  probe.dataset.counts.validation=1000 probe.dataset.counts.test=1000 \
  "probe.dataset.shard_size=$set_probe_shard" \
  "hydra.run.dir=$set_probe_runs/capture" "wandb.name=$set_probe_run-capture")
set_probe_capture_job=${set_probe_capture_job%%;*}

set_probe_train_job=$(sbatch --parsable "${set_probe_log_args[@]}" \
  "--dependency=afterok:$set_probe_capture_job" \
  --cpus-per-task=12 --mem=64G "--time=$set_probe_train_time" \
  scripts/cluster/isambard/train_probe.slurm \
  "${set_probe_common[@]}" "probe.training.num_steps=$set_probe_steps" \
  probe.training.batch_size=128 probe.training.control=none \
  probe.training.weight_decay=0.003 \
  "hydra.run.dir=$set_probe_runs/train" "wandb.name=$set_probe_run-train")
set_probe_train_job=${set_probe_train_job%%;*}

set_probe_eval_job=$(sbatch --parsable "${set_probe_log_args[@]}" \
  "--dependency=afterok:$set_probe_train_job" \
  --cpus-per-task=12 --mem=32G "--time=$set_probe_eval_time" \
  scripts/cluster/isambard/eval_probe.slurm \
  "${set_probe_common[@]}" \
  "probe.evaluation.checkpoint=$set_probe_runs/train/checkpoints/best.pt" \
  probe.evaluation.split=test \
  "hydra.run.dir=$set_probe_runs/eval" "wandb.name=$set_probe_run-eval")
set_probe_eval_job=${set_probe_eval_job%%;*}

printf 'Capture: %s\nTrain: %s\nEval: %s\nDataset: %s\nRuns: %s\n' \
  "$set_probe_capture_job" "$set_probe_train_job" "$set_probe_eval_job" \
  "$set_probe_dataset" "$set_probe_runs"
BASH
```

For **500,000 training / 1,000 validation / 1,000 test episodes and 100,000
updates**, change these values near the top of the same block before submitting:

```bash
set_probe_train_count=500000
set_probe_steps=100000
set_probe_shard=16384
set_probe_capture_time=02:00:00
set_probe_capture_mem=64G
set_probe_train_time=04:00:00
set_probe_eval_time=01:00:00
```

These are initial resource allocations, **not measured completion estimates**.
Rebenchmark this decoder before tightening them. A 16,384-episode shard holds
about 8.17 GiB of arrays at reference width; capture needs room for buffers and
write copies. The entire 502,000-episode dataset holds about **250.28 GiB of
arrays on disk**, plus file overhead. Keep separate room for rolling
`best.pt`/`last.pt`, atomic checkpoint writes, artifact downloads and caches.
Scratch quota and host RAM are different limits. Training reads mmap batches;
it does not allocate 250 GiB of RAM.

Capture resume reuses compatible completed shards. To recover an interrupted
chain, resubmit capture with the **same dataset path and capture settings**, then
submit new `afterok` dependencies on the new job IDs. For an interrupted training
run, also pass `probe.training.resume=<train-dir>/checkpoints/last.pt` with the
same training settings and update budget. A dependency on a failed job does not
become successful when a different job resumes its work. Use a fresh evaluation
directory if an earlier evaluation already wrote results.

The probe checkpoint belongs to the decoder; `probe.capture.checkpoint` refers
to the frozen GDN used to generate the dataset. To fit the constant or shuffled
control, submit another training/evaluation pair on the same captured dataset,
changing `probe.training.control` to `constant` or `shuffled_targets` and giving
both jobs distinct run directories and W&B names. Evaluation restores the
control kind from its checkpoint. Test episodes are never used for selection.

## Full-decoder benchmark

Capture a bounded dataset and measure complete decoder updates in one job:

```bash
: "${SCRATCHDIR:?SCRATCHDIR must be set}"
set_probe_root="$SCRATCHDIR/iccl-analysis"
set_probe_benchmark="set-m4t7d32-benchmark-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$set_probe_root/logs/slurm"
sbatch --mem=64G --time=01:00:00 \
  "--output=$set_probe_root/logs/slurm/%x_%j.out" \
  "--error=$set_probe_root/logs/slurm/%x_%j.err" \
  scripts/cluster/isambard/benchmark_probe.slurm \
  probe.capture.checkpoint=/path/to/reference-gdn/checkpoint.pt \
  "probe.dataset.path=$set_probe_root/module-decoder/datasets/$set_probe_benchmark" \
  "probe.solver.cache_dir=$set_probe_root/cache/probe-assignment" \
  probe.dataset.counts.train=512 \
  probe.dataset.counts.validation=64 \
  probe.dataset.counts.test=64 \
  probe.benchmark.capture_first=true \
  "hydra.run.dir=$set_probe_root/module-decoder/runs/$set_probe_benchmark" \
  "wandb.name=$set_probe_benchmark"
```

The default benchmark uses the full decoder and batch size 128. Its
`benchmark.json` records capture time and end-to-end update timings, including
the exact assignment solver, backward pass and optimizer.
