# Isambard GPU jobs

Submit these scripts from the repository root. Each job requests one GPU on one
node, runs one Python process, and uses the site's default partition and account.
Isambard allocates a GH200 Superchip with its associated host memory per GPU;
the scripts set CPU counts for their worker/thread budgets. See
[Isambard's batch-job guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm/#running-a-single-batch-job).

| Job | Script | CPUs per task | Walltime |
| --- | --- | ---: | ---: |
| GDN training | `train.slurm` | 8 | 12 hours |
| GDN evaluation | `eval.slurm` | 4 | 4 hours |
| Probe dataset capture | `capture_probe.slurm` | 4 | 4 hours |
| Probe training | `train_probe.slurm` | 12 | 12 hours |
| Probe evaluation | `eval_probe.slurm` | 12 | 4 hours |
| Probe benchmark | `benchmark_probe.slurm` | 12 | 1 hour |

All launchers delegate to `run.sh`, which initializes the module command if
needed, loads `cudatoolkit`, then exports `CC=/usr/bin/gcc-12` and
`CXX=/usr/bin/g++-12`. It checks the compiler commands before starting Python,
records their versions, the Git commit and GPU information in the job log, and
uses `uv run --locked`. Add any additional module loads before those compiler
exports; future Isambard launchers should also delegate to this runner.

W&B defaults to online. Set `WANDB_MODE=offline` or `WANDB_MODE=disabled` when
submitting to select another mode. Model/data/probe settings are ordinary Hydra
arguments after the script name. Scheduler overrides such as `--time`,
`--cpus-per-task`, `--partition` or `--account` go before the script name. Set
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
uv run --locked python scripts/make_eval_sets.py
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

Capture final memory states from a GDN checkpoint into a reusable dataset:

```bash
sbatch scripts/cluster/isambard/capture_probe.slurm \
  probe.capture.checkpoint=/path/to/gdn/checkpoint.pt \
  probe.dataset.path=outputs/module-decoder/datasets/reference
```

After capture completes, train a probe on that dataset:

```bash
sbatch scripts/cluster/isambard/train_probe.slurm \
  probe.dataset.path=outputs/module-decoder/datasets/reference
```

Evaluate the resulting probe checkpoint on the held-out test split:

```bash
sbatch scripts/cluster/isambard/eval_probe.slurm \
  probe.dataset.path=outputs/module-decoder/datasets/reference \
  probe.evaluation.checkpoint=/path/to/probe/checkpoints/best.pt \
  probe.evaluation.split=test
```

The probe checkpoint belongs to the decoder; `probe.capture.checkpoint` refers
to the frozen GDN used to generate the dataset. See the
[module-decoder guide](../../../docs/module-decoder.md) for controls,
resumption, figures and output formats.

## Full-decoder benchmark

Capture a bounded dataset and measure complete decoder updates in one job:

```bash
sbatch scripts/cluster/isambard/benchmark_probe.slurm \
  probe.capture.checkpoint=/path/to/reference-gdn/checkpoint.pt \
  probe.dataset.path=outputs/module-decoder/datasets/reference-benchmark \
  probe.dataset.counts.train=512 \
  probe.dataset.counts.validation=64 \
  probe.dataset.counts.test=64 \
  probe.benchmark.capture_first=true
```

The default benchmark uses the full decoder and batch size 128. Its
`benchmark.json` records capture time and end-to-end update timings, including
the exact assignment solver, backward pass and optimizer.
