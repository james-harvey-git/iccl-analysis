# iccl-analysis

In-context continual learning (ICCL) in linear-attention models: synthetic dataset generation, model training, and mechanistic interpretability analysis of the meta-learned ICCL algorithm.

DPhil rotation project. The synthetic dataset is adapted from [Redhardt, Akram & Schug (2025), "Scaling can lead to compositional generalization"](https://arxiv.org/abs/2507.07207).

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
suites, and the paired-position/constituent-rehearsal diagnostics. The canonical
capability cells are also used for train-time monitoring; diagnostics run only
during standalone evaluation. Data and backups are gitignored.

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
`evaluation.suites=all` (default), `capabilities`, or `retention_position`:

```bash
uv run python scripts/eval.py \
  'evaluation.checkpoints=[outputs/<date>/<time>/checkpoints/best.pt]' \
  evaluation.suites=all wandb.mode=online
```

Complete numerical results and metadata are saved locally under the evaluation
run's `evaluation-results/` and uploaded when W&B is enabled. Reconstruct figures
with `uv run python scripts/plot_evaluation.py <evaluation-results-dir>
--out-dir outputs/evaluation-plots`.
