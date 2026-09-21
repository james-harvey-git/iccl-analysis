# iccl-analysis

In-context continual learning (ICCL) in linear-attention models: synthetic dataset generation, model training, and mechanistic interpretability analysis of the meta-learned ICCL algorithm.

DPhil rotation project. The synthetic dataset is adapted from [Redhardt, Akram & Schug (2025), "Scaling can lead to compositional generalization"](https://arxiv.org/abs/2507.07207).

Exploratory work in `notebooks/` is gitignored. Reusable analyses belong in
`src/iccl/analysis/`; a notebook chosen to demonstrate established results can
be included deliberately with `git add -f notebooks/<name>.ipynb`.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

Experiments are dispatched with [Hydra](https://hydra.cc/); configs live in `configs/`.

```bash
uv run python scripts/make_eval_sets.py
uv run python scripts/train.py
```

## Frozen evaluation data

`data/eval_sets/` is the authoritative bundle. It contains training-distribution
validation, the fixed-D canonical/task-variation/module-variation capability
suites. Retention uses 256 independent worlds per physical cell, each evaluated
at every delay in repeat/shared/unexposed histories. At M=T=8, D=32 this is 2,048
episodes per condition, each with eight history tasks and one final task.
Train-time monitoring selects 256 canonical episodes per condition from these
same frozen data: one delay per world, balanced to 32 worlds per delay. Generation
and monitoring selection are deterministic and fixed across checkpoints. Data and backups are gitignored.

Prepare the bundle once with `scripts/make_eval_sets.py`, using the same data and
seed overrides as training. Repeating the command verifies and reuses a matching
bundle. If replacement is needed, generation completes before the old directory
(including any obsolete nested versions) is moved to
`outputs/eval-set-backups/<name>-<timestamp>/`. A failed generation leaves the
active bundle untouched. Do not rebuild it while another run is loading it.

The manifest records generation settings and file checksums. Training and
evaluation reject missing, altered or mismatched bundles; neither silently
regenerates them. Training duration, optimizer/model settings, W&B settings and
bootstrap settings do not affect frozen-data identity. Data or seed overrides
must match those used to prepare the bundle, including the validation distribution.

All evaluation calls use the same directory. Select which suites to score with
`evaluation.suites=all` (default), `capabilities`, or `retention_position`. The last
option selects the full canonical retention evaluation from the same bundle;
it does not generate a second position-diagnostic population:

```bash
uv run python scripts/eval.py \
  'evaluation.checkpoints=[outputs/<date>/<time>/checkpoints/best.pt]' \
  evaluation.suites=all wandb.mode=online
```

Complete numerical results and metadata are saved locally under the evaluation
run's `evaluation-results/` and uploaded when W&B is enabled. Set
`evaluation.results_dir=outputs/shared-evaluation-results` to use a shared cache
across full/standalone runs. Matching cached suites are reusable; missing suites,
changed weights/data/numerical settings or monitor-only results are rejected.
Use a fresh output directory for incompatible or incomplete caches. Reconstruct figures
with `uv run python scripts/plotting/plot_evaluation.py <evaluation-results-dir>
--out-dir outputs/evaluation-plots`.

Paper plotting scripts also live in `scripts/plotting/`. Render the reference
learning, retention and composition panels from an evaluation step directory,
or redraw the cached retention trajectory as separate paper panels:

```bash
uv run python scripts/plotting/plot_learning_curves.py --results PATH/TO/step_2100000
uv run python scripts/plotting/plot_retention_learning.py --results PATH/TO/step_2100000
uv run python scripts/plotting/plot_composition_learning.py --results PATH/TO/step_2100000
uv run python scripts/plotting/plot_retention_trajectory.py --mode plot --paper
```

These commands write PDF/PNG figures under `outputs/`. The retention-learning
script overlays original, exact-repeat, shared and unexposed curves and exports a separate
mean-savings panel using the saved paired bootstrap intervals. It also verifies
the decomposition across all supplied configurations and writes an audit CSV.
The learning and trajectory scripts additionally provide LaTeX subfigure snippets.
The composition script exports exposed, unexposed and no-history learning curves
alongside the paired unexposed-minus-exposed benefit, preserving its saved intervals.


### Retention controls and uncertainty

Every matched history has the same fresh final examples and target mixture.
Only the designated original encounter changes: exact repeat, the same module
pair with different weights, or a disjoint replacement (unexposed). Neither target
module appears elsewhere in standard retention history. The unaffected background
covers and connects all M−2 other modules. The full episode need not be connected.

At each final demonstration, total savings = unexposed − repeat, module savings =
unexposed − shared, and episodic savings = shared − repeat. Total is exactly the
sum of the other two. Binary weighting omits shared and reports total only.
Full evaluations bootstrap whole worlds with every delay retained; monitoring
uses independent worlds within delay strata and weights delays equally. Counts
on plots denote independent worlds, not the number of serialized episodes.
Original learning is taken from the exact-repeat episode's original encounter.
Delay also changes that encounter's serial position, so these curves do not
identify an isolated forgetting rate or a particular memory mechanism.

Live W&B monitoring includes the four learning panels and three delay panels:
unexposed error, exact-repeat error and total savings. Each panel logs the current
curve at the source training step; W&B's step slider shows progression. Local
`monitor/step_*.json` files preserve curve estimates, intervals and protocol
metadata even with W&B disabled; the compact mean arrays remain in `.npz` files.

### Optional controlled rehearsal

To include this separate experiment when preparing a bundle, pass
`data.eval_sets.rehearsal.enabled=true`. Use the same setting when validating that
bundle and select `evaluation.suites=rehearsal` to evaluate it. Standard `all`,
`capabilities` and `retention_position` selections exclude rehearsal.

The final task stays identical across all three encounter conditions and all
none/one/both rehearsal modes. Later rehearsal is identical across the three
histories. Neither target module occurs before the original encounter. Rehearsal
exposes each selected constituent once with a non-target partner, never as the
target pair. The control is therefore **unexposed at the original encounter**,
not globally unexposed when rehearsal occurs. A connected unaffected background
requires T≥M and at least two intervening tasks; unsupported cells fail explicitly.

### Protocol migration

Regenerate frozen bundles and numerical evaluations for this protocol; historical
novel-final-task results cannot be relabelled as unexposed-history results. Old
checkpoint weights remain usable with the current explicit evaluation configuration.
`eval_sets.retention.num_worlds` owns full retention sampling;
`eval_sets.retention.monitor_num_sequences` controls only the live subset.
`eval_sets.num_sequences` still controls ICL/composition and training validation.
The obsolete `retention.position_diagnostic.num_worlds` setting is rejected.
Training data generation and its golden-stream checksums are unchanged. W&B jobs
record the metric protocol and start fresh reporting runs; they do not resume
historical metric series with different definitions.
