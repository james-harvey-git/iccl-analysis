# Fixed-world task-vector probes

The experiment asks whether a frozen GDN makes the current weighted task vector
linearly accessible **within a fixed teacher world**. Each world has independent
teacher modules and its own fitted probes. World coordinates are never pooled.
The design and acceptance criteria are in [issue #29](https://github.com/james-harvey-git/iccl-analysis/issues/29).
Graph-structure analysis is a separate, subsequent investigation.

## Experimental design

The default world has eight modules, nine tasks and 32 demonstrations per task.
Each task activates two distinct modules, with coefficients sampled independently
from {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}. Modules, biases and readout stay fixed across
that world's trials. Trials independently redraw coefficients, inputs and task
orders.

The 28 support pairs are partitioned into 16 training, six validation and six
unseen-test pairs. Each partition covers every module. Training supports are
connected and each module occurs in at least two training pairs, ensuring that
excluding the current target support still permits full background coverage.
Background tasks use training supports, exclude the target pair, and collectively
cover every module. A fresh random order places the target at the configured task
position. Targets cycle through independently shuffled support lists, balancing
support coverage without conditioning on model activations or test scores.

There are four independent stream populations per world and target position:
training, validation, familiar-support test and unseen-support test. Familiar
tests use training supports with fresh streams. All capture points, controls and
representations for a trial stay in the same split. Development and confirmation
use disjoint RNG namespaces; confirmation requires an explicit protocol-freeze
flag. Freeze budgets, feature choices, exposure bins and coverage thresholds on
development worlds before running fresh confirmation worlds.

The only capture points are:

| Point | Available information |
| --- | --- |
| `final_input` | D−1 completed demonstrations and the final unanswered input |
| `final_answer` | All D demonstrations, after processing the final answer |
| `branch_input` | A fresh unanswered query from a copy of the complete post-answer cache |

The branch copies fast-weight matrices **and convolution history**, occurs
before the next task boundary, and is discarded. Model predictions are retained
at the final ordinary input and at the branch input.

Capture stores residuals after each mixer and after each MLP, SwiGLU hidden
activations before the down projection, fast-weight state matrices, and native
head reads before output normalization/gating. Views select all layers, individual
layers, individual state/read heads, or named groups chosen before results.
Flattening preserves all configured coordinates; there is no dimensionality
reduction. Capture on CUDA uses segmented FLA terminal states rather than full
per-token state trajectories.

Each world × task position × capture point × feature selection has a separate
affine ridge probe. Feature centering/scaling uses training moments only, constant
features get scale one, and the intercept is unpenalized:

```text
minimize ||X_standardized W + b − Z||²_F / n + lambda ||W||²_F
```

Only converged candidates with defined **full-validation macro coordinate R²**
can be selected. Ties choose the larger lambda. Tests never select lambda.
Small problems use a direct primal/dual solve; larger problems use streamed
conjugate gradients without forming a feature covariance matrix.

Controls are a training target mean, shuffled training targets, an untrained GDN
on the same full prefixes, and raw current-task observations. Shuffling is within
world and position, shared across all capture points and features. The observation
control includes only available x/y values and the unanswered query; it does not
introduce nonlinear x/y products.

Evaluation computes macro coordinate R², exact top-two support accuracy and
active-coefficient RMSE separately within worlds, then averages worlds equally.
A constant test coordinate makes macro R² undefined and is explicitly recorded.
Top-two ties use coordinate index order. Bootstrap intervals resample whole
worlds; comparisons with controls are paired by world ID.

For prior exposure, count preceding tasks containing each active target constituent,
sort the two counts, and plot `(e_min, e_max)`. Current-task demonstrations,
later tasks and the discarded query do not count. Position 1 only has (0,0);
the final position has both constituents exposed; their total prior exposure
cannot exceed position−1 because backgrounds exclude the target support.
The three-by-three default heatmap uses position rows and capture-point columns.
Its companion fixed-scale vector MSE divides each coordinate's squared error by
that coordinate's variance in the original full training set. It keeps the same
normalizer across cells, captures and controls. Supported cells in a panel use a
common world cohort. Counts, target-pair coverage and coordinate variances are
saved, and impossible/undersampled cells remain distinct. These are observational
associations conditional on task position, not causal history interventions.

## Three stages and configuration

`configs/probe.yaml` selects `probe=task_vector`. The nested sections are
`probe.dataset`, `probe.training` and `probe.evaluation`. A resolved configuration
is saved with each artifact. These examples use a **small smoke preset**:

```bash
uv run python scripts/capture_probe_dataset.py probe=smoke \
  experiment.root=outputs/task-vector/smoke \
  probe.dataset.checkpoint=/absolute/path/to/gdn.pt
uv run python scripts/train_probe.py probe=smoke \
  experiment.root=outputs/task-vector/smoke
uv run python scripts/eval_probe.py probe=smoke \
  experiment.root=outputs/task-vector/smoke
```

Repeat the same scientific overrides at all three stages. Stage two checks the
capture manifest and reads only train/validation; stage three checks the selected
probe identity and reads only the configured test populations. The checkpoint is
needed only for capture. It supplies architecture, teacher dimensions, biases,
scaling and boundary convention. The experimental module count and weighting are
explicitly configured.

Useful settings:

| Setting | Purpose |
| --- | --- |
| `dataset.num_worlds`, `modules_per_world`, `tasks_per_episode`, `demos_per_task` | Sampling budgets/design |
| `dataset.trajectories` | Trial counts **per world and target position**, for all four splits |
| `dataset.capture.representations` | Families and all/layer/head views |
| `dataset.capture.head_groups` | Named lists of one-based [layer, head] pairs |
| `training.representations` | All captured views or a configured subset |
| `training.l2_grid` | Positive regularization candidates |
| `training.solver` | Direct/CG choice, precision, memory bound and convergence criteria |
| `training.sample_sizes` | Development-only nested budgets; full budget always included |
| `training.monitoring` | Fixed monitor subsets, metric cadence and CG checkpoint cadence |
| `evaluation.exposure` | Predeclared bins and minimum trial/world coverage |

The default capture can occupy hundreds of GiB for a full checkpoint. Capture
prints a feature-storage estimate before processing trials. `probe=pilot` keeps
the full checkpoint dimensions and D=32 but uses one world and 64/32/32/32 trials.
`probe=smoke` uses D=4 and two worlds. Neither is a confirmation experiment.

## Monitoring, artifacts and resume

All stages share one W&B run ID in `experiment.root/wandb-run.json`. Only the
coordinator writes to W&B; workers append durable JSONL events on shared storage.
Live training figures show actual train/validation subset MSE and R² versus
solver iteration for the current fit on each worker. Direct solves report
initial/final values; they do not manufacture an iterative training history.
Full-validation selection scores, solve residuals and wall times remain in
candidate records. Development data-budget curves use validation scores.

W&B evaluation keys are grouped by phase and show world-averaged position curves,
layer/head views, paired control differences, exposure heatmaps and model query
error. Detailed rows are in a table; the default figures do not draw a line for
every world. Final compact result artifacts include definitions and provenance.
Bulk captures and probe weights stay on scratch.

`wandb.mode=offline` or `disabled` keeps local event streams without starting a
W&B writer. To replay those events and publish saved figures to the same run:

```bash
uv run python scripts/report_probe.py probe=smoke \
  experiment.root=outputs/task-vector/smoke wandb.mode=online
```

Online reporting stores consumed event offsets and a monotonic W&B step. It is
best-effort telemetry; the immutable numerical artifacts and local events remain
the scientific record if network delivery is interrupted. The study writer lock
prevents simultaneous stage/report writers.

Capture fingerprints the checkpoint, sampler, architecture and capture contract.
Atomic shards include checksums and deterministic world/split/position/trial
identifiers. Resume validates existing shards rather than silently overwriting
them. Training fingerprints its input dataset and fitting settings, resumes
partial CG iterates, and reuses verified completed candidates/selected probes.
Incompatible configurations require fresh output directories. Old module-decoder
artifacts are deliberately incompatible.

The results folder contains `summary.json`, `report.md`, per-world scores,
held-out predictions, interactive HTML figures, and PNG/PDF exposure figures.
Regeneration needs no model inference or refitting:

```bash
uv run python scripts/plotting/plot_probe.py \
  --results outputs/task-vector/smoke/development/results/summary.json \
  --out-dir outputs/task-vector/smoke/development/results/plots
```

## Isambard and bounded pilots

See the [Isambard guide](../scripts/cluster/isambard/README.md). Workers own
disjoint worlds, with one visible GPU each; there is no DDP. The submission helper
limits the study to eight GPUs and queues capture → fit → evaluation with
`afterok` dependencies. Explicit memory is **per node**, not per worker.

The bounded benchmark refuses an existing dataset and limits worlds/trials.
It measures all three stages and repeated direct/CG fits of full state and MLP
views, separating warmed solve timings from cold pipeline compilation/I/O:

```bash
uv run python scripts/benchmark_probe.py probe=pilot \
  experiment.root=outputs/task-vector/pilot-unique \
  probe.dataset.checkpoint=/absolute/path/to/gdn.pt
```

Measure a representative larger training budget separately before extrapolating
CG convergence or choosing confirmation budgets. Check numerical convergence and
capture parity before interpreting throughput or decoding performance.
