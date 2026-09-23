# Linear module decoding from terminal GDN memory

This pipeline implements the [approved experiment](https://github.com/james-harvey-git/iccl-analysis/issues/27).
It tests linear recovery of an equivalent teacher representation. Successful
decoding does not establish disjoint storage subspaces or causal use of modules.

## Configuration and scientific contract

All four Hydra entry points use `configs/probe.yaml`, composing
`configs/probe/module_decoder.yaml`. `probe=smoke` reduces counts, batches and
updates while retaining the complete decoder implied by the GDN checkpoint.
The checkpoint and dataset path are launch inputs. Counts/rates are pilot settings.

Episodes use the production constructive curriculum: M=T=8, D=32, two modules
per task, one surplus task, coverage/connectivity required, full latent rank not
required. Teachers have dimensions 16→16→16, biases and discrete coefficients.
One ordinary boundary is appended after the final y-token, giving 521 tokens.
The append is local to capture; production sequence serialization is unchanged.

Capture reconstructs the frozen GDN architecture from its checkpoint. Final
states are flattened in layer, head, value, key order **after** the appended
boundary. No trajectories or residual activations are retained. `backend=auto`
uses FLA on CUDA and the reference recurrence on CPU/MPS. Backend, device,
precision, versions and source hashes are recorded; state storage is FP32.
CPU/MPS operation does not import FLA. No GDN backward pass is required.
The model uses `capture_final=True`; existing `capture=True` behavior is preserved.

The reference checkpoint gives 131,072 inputs. A single affine layer produces
4,608 outputs: `[8,2,17,16]` augmented module matrices and a shared `[16,16]`
readout. This is **603,984,384 parameters**, without a rank constraint or OOM
fallback. Other GDN checkpoints determine other input widths, retaining the same
teacher/target dimensions.

Targets contain unweighted module content. Canonicalization multiplies each
module column and bias by its readout-row norm, then normalizes that readout row.
Intermediates use float64; stored targets use float32. Zero/nonfinite norms fail.

The exact `branch_warm` matcher chooses eight task-pair swaps and **one shared**
hidden-neuron permutation, including the readout. It builds float64 costs from
detached FP32 predictions on CPU. Targets are gathered on the training device;
FP32 squared error on the original predictions supplies decoder gradients.
`readout_weight=1` gives equal weight per scalar; the denominator remains 4,608
under a different readout weight.

## Setup and capture

Use `uv sync`. Matching needs a C++17 compiler: Apple Command Line Tools on macOS,
or GCC/Clang on Linux. `CXX=clang++` overrides `c++`. Compilation is lazy with
`-O3 -std=c++17 -pthread -fPIC`, without fast-math. A concurrency-safe cache under
`outputs/.cache/probe-assignment/` is keyed by source/compiler/platform identity.
Ordinary model imports/training do not compile it.

From the repository root, replace the source path with the chosen GDN:

```bash
uv run python scripts/capture_probe_dataset.py \
  probe.capture.checkpoint=/path/to/gdn/checkpoint.pt \
  probe.dataset.path=outputs/module-decoder/datasets/reference-pilot
```

Alternatively, use a W&B checkpoint:

```bash
uv run python scripts/capture_probe_dataset.py \
  'probe.capture.checkpoint=wandb://ENTITY/PROJECT/ARTIFACT:ALIAS' \
  probe.dataset.path=outputs/module-decoder/datasets/reference-pilot
```

Defaults are 10,000 train / 1,000 validation / 1,000 test episodes. Reference-width
arrays need about 6.2 GiB before filesystem overhead; capture prints an estimate.
Only capture needs counts, dataset seed and source GDN. Consumers read the manifest.

Worlds use independent, indexed CPU Philox streams in a stable uint64 namespace,
separate from current production training/evaluation seed ranges. Worker count,
shard size and generation order do not change episode worlds. Stored metadata
includes seeds/indices, coefficients, module IDs, first appearances, occurrence
counts, latent rank, original teachers and canonical targets.

Repeat capture to validate/reuse completed shards and finish interrupted work.
Compatible training counts can grow via `probe.dataset.counts.train=20000`;
validation/test populations remain fixed. Different checkpoints, numerical
settings or generation/capture source hashes require a fresh dataset path.
One writer holds the capture lock. Shards publish atomically with checksums and
recoverable manifest updates. Readers reject incomplete/corrupt requested splits.
Batch reads group file access by shard and restore the requested episode order.
Each reader process retains at most 32 mapped shards; rows are copied before
eviction, and shuffled controls retain complete target-episode pairings.

## Train, resume and controls

```bash
uv run python scripts/train_probe.py \
  probe.dataset.path=outputs/module-decoder/datasets/reference-pilot \
  hydra.run.dir=outputs/module-decoder/runs/reference-pilot

uv run python scripts/train_probe.py \
  probe.dataset.path=outputs/module-decoder/datasets/reference-pilot \
  probe.training.resume=outputs/module-decoder/runs/reference-pilot/checkpoints/last.pt \
  hydra.run.dir=outputs/module-decoder/runs/reference-pilot
```

Repeat nondefault training overrides on resume. Compatibility includes shard
identities, seed, batch size, optimizer, schedule/update budget, objective,
precision and numerical implementation. Worker/solver thread counts, logging/
checkpoint cadence and output directory may change. The saved cursor counts
**consumed** samples, never prefetched samples. Adam, scheduler and Python/NumPy/
Torch RNG states continue with the decoder. Extending a captured training split
requires a new probe run, since it changes the sample population.

Training reports the first completed update and the configured logging cadence.
Metrics are flushed to W&B at the end of the same step, after merging any
validation result. `probe/train/seconds_per_update` includes batch loading and
optimization; `probe/train/data_wait_seconds` isolates waiting for that batch.

`last.pt` and validation-selected `best.pt` are both resumable. Keep the pair
together when moving runs. A prior selected best is carried into a new resume
directory. If a newer best replaced the historical best associated with an older
last checkpoint, resume that best or restore the matching pair. Test worlds never
influence fitting, scheduling or selection.

Fit the state-independent control as a separate run:

```bash
uv run python scripts/train_probe.py \
  probe.dataset.path=outputs/module-decoder/datasets/reference-pilot \
  probe.training.control=constant \
  hydra.run.dir=outputs/module-decoder/runs/constant-pilot
```

It trains a nonzero-initialized 4,608-vector with the same matching loss and its
own validation selection. It measures what target structure and alignment can
explain without state information. The optional `probe.training.control=shuffled_targets`
setting instead trains a full decoder with a recorded deterministic
derangement of **whole training episodes' targets**, including internal metadata.
States stay in place; held-out worlds stay correctly paired. Every evaluation
also includes a zero-output baseline.

## Evaluate and redraw

```bash
uv run python scripts/eval_probe.py \
  probe.dataset.path=outputs/module-decoder/datasets/reference-pilot \
  probe.evaluation.checkpoint=outputs/module-decoder/runs/reference-pilot/checkpoints/best.pt \
  probe.evaluation.results_dir=outputs/module-decoder/results/reference-pilot

uv run python scripts/eval_probe.py \
  probe.dataset.path=outputs/module-decoder/datasets/reference-pilot \
  probe.evaluation.checkpoint=outputs/module-decoder/runs/constant-pilot/checkpoints/best.pt \
  probe.evaluation.results_dir=outputs/module-decoder/results/constant-pilot

uv run python scripts/plotting/plot_probe.py \
  --results outputs/module-decoder/results/reference-pilot \
            outputs/module-decoder/results/constant-pilot \
  --histories outputs/module-decoder/runs/reference-pilot/history.json \
              outputs/module-decoder/runs/constant-pilot/history.json \
  --out-dir outputs/module-decoder/plots/reference-pilot
```

Evaluation restores decoder kind and loss weight from the probe checkpoint. It
needs no source GDN or capability bundle. Use a fresh results directory. The
saved plotting path checks checksums and comparison identities. Evaluation
precision defaults to auto (BF16 CUDA, FP32 otherwise).

Reports retain chosen masks/permutations, joint/component errors, errors by task
position, repeated-module consistency and latent-rank strata with counts. The
primary aggregate includes all sampled episodes. Confidence intervals resample
whole episodes, retaining within-episode dependence. Intervals are omitted when
fewer than two episodes or two bootstrap replicates are available.

With `wandb.mode=online`, standalone evaluation also reports six interactive
Plotly panels under `probe/test/figures/` (or `probe/validation/figures/`):

- Module reconstruction error by task position, with confidence intervals.
- Functional reconstruction MSE and nMSE by task position, with confidence intervals.
- Weight, bias and readout reconstruction errors, with confidence intervals.
- Cumulative distributions of episode parameter and functional errors.
- Reconstruction errors by observed latent rank, with group sizes and confidence intervals.
- Repeated-module consistency versus reconstruction error for individual episodes.

Each panel includes the zero-output baseline and identifies the evaluated decoder
or control. Hover labels expose exact estimates, intervals or episode IDs as
appropriate. The matching `probe/<split>/summary` table includes all position and
rank estimates, interval bounds, episode counts and floored-variance counts.
All-population scalar means are logged under `probe/<split>/{decoder,zero}/`
at the evaluated checkpoint's training step. These dashboard outputs use the
saved measurements and intervals; they do not refit or realign predictions.
`wandb.mode=offline` records the same media locally for later sync.
Dashboard figures and the table do not require `wandb.upload_results=true`;
that flag controls the full results artifact separately.

Consistency is within-module occurrence variance, equally averaged over repeated
module IDs in the common predicted hidden basis. Zero output is perfectly
consistent, so interpret this metric alongside reconstruction accuracy.

**Oracle-coefficient functional reconstruction** uses fresh deterministic uniform
`[-1,1]` inputs, coefficients corresponding to the chosen pair swaps, `1/sqrt(2)`
composition and the predicted readout. It compares against the original teacher.
nMSE divides task MSE by mean output variance over these fresh inputs, floored at
the project's `1e-12`; floored-task counts are reported. Functional errors do not
change the alignment or enter training. This does not measure coefficient recovery.

## Full-decoder cluster benchmark

On Isambard, use the dedicated
[job scripts](../scripts/cluster/isambard/README.md), including
`scripts/cluster/isambard/benchmark_probe.slurm`. They load CUDA and select the
tested GCC/G++ host compilers automatically before running Python.

Capture a small real dataset, then measure the reference-width decoder at batch
size 128. Launch from the repository root:

```bash
mkdir -p logs/slurm
sbatch scripts/cluster/probe.slurm benchmark \
  probe.capture.checkpoint=/path/to/reference-gdn/checkpoint.pt \
  probe.dataset.path=outputs/module-decoder/datasets/reference-benchmark \
  probe.dataset.counts.train=512 \
  probe.dataset.counts.validation=64 \
  probe.dataset.counts.test=64 \
  probe.benchmark.capture_first=true
```

The launcher also accepts `capture`, `train` and `eval`. It requests one GPU,
12 CPUs and 64 GiB host memory; machine settings live in SLURM directives.
It does not prepare capability bundles. `WANDB_MODE=disabled sbatch ...` disables
its default online reporting. A local operational smoke is:

```bash
uv run python scripts/benchmark_probe.py probe=smoke \
  probe.capture.checkpoint=/path/to/gdn/checkpoint.pt \
  probe.dataset.path=outputs/module-decoder/datasets/smoke \
  probe.benchmark.capture_first=true
```

Smoke retains all 603,984,384 parameters for the reference checkpoint. Tests use
an explicitly tiny compatible checkpoint; those timings are not reference/H100
performance measurements.

`benchmark.json` separates warmup, measured and profiled updates (defaults
10/100/10). All are real optimization updates; warmup advances Adam/model state.
They include loading, transfers, decoder, native exact matching, differentiable
loss, backward, finite-gradient checks and Adam. `probe.training.resume` can benchmark
trained weights with matching training settings. Constant benchmarks are rejected.
`probe.benchmark.profile_steps=0` disables stage instrumentation.

Measured updates synchronize at their boundaries. Profiling additionally
synchronizes between stages, changing throughput. Native cost/search times are
summed CPU worker times that overlap; do not add them to wall-time stages.
Per-update losses, gradient norms, search counters, median/p95/p99 timings and
step ranges support early/later comparisons. Checksum verification warms the
filesystem cache before timing; OS caches are not evicted.

GPU allocation/reservation peaks, parent-process peak RSS (excluding loader
workers), versions, compiler flags, CPU allocation and storage context are
recorded. Capture and setup/build/load are separate costs. Validation, checkpoint
writes and control fitting are excluded from timed updates; the benchmark writes
no large optimizer checkpoints.

FP32 reference weights need about 2.25 GiB; weights, gradients and two Adam moments
need about 9.00 GiB before autocast copies, temporaries and allocator overhead.
These are planning estimates. GPU feasibility must be measured on the cluster.
OOMs never silently shrink the decoder.

## Artifacts and checks

Datasets contain `manifest.json` and split `shard_*/` directories with checksummed
typed `.npy` arrays. Runs contain configuration, `history.json`, `training.json`,
plots and `checkpoints/{last,best}.pt`. Evaluations contain `manifest.json`,
`summary.json`, `episodes.npz` and plots. Generated artifacts/native libraries live
under ignored `outputs/`; no scratchpad file is a runtime dependency.

Scalars and dashboard figures respect `wandb.mode`. `wandb.upload_weights=true`
opts into a weights-only probe upload; `wandb.upload_results=true` opts into the
full numerical results bundle and saved figures. Captured
datasets and resumable Adam checkpoints stay local.

```bash
uv run pytest
uv run ruff check
uv run pyright
# On CUDA hardware with the locked FLA dependencies:
uv run pytest -m cuda tests/test_ops_parity.py
```

CUDA parity tests check predictions and terminal states, including lengths around
chunk boundaries and 521 tokens. They need actual CUDA hardware. Installation and
tests do not submit cluster jobs.
