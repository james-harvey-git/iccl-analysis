# Task-vector scale calibration

The pipeline is ready for the eight-world **development** study. Calibration supports
a 4 GiB direct-solver allowance, 1,024 training streams per world/position, a wider
ridge-penalty grid, and 2,048 streams in each held-out population. The full study
and fresh confirmation worlds have not been launched by this calibration.

## Question and logic

The first pilot used 64 training streams. It verified the pipeline but could not
predict the cost or convergence of a 1,024-stream full-state probe. This calibration
compares direct and iterative ridge solutions on identical captured data, measures
validation learning curves, and checks whether the exposure figures have enough
trials and worlds per cell. Penalties are assessed using validation data only.

## Setup and definitions

The frozen source model is step 2,900,000 of W&B run `0bdfkn9e`, artifact
`weights-variable-surplus-s0-4-400k:v14`. One development world has eight modules,
two active constituents per task, nine tasks and 32 demonstrations per task.
The target is task 9. The inputs are all MLP hidden activations (4,096 features)
and all fast-weight states (131,072 features), at final input, final answer and
discarded branch input. All configured controls are retained.

The GPU dataset has 1,024 training, 512 validation, 512 familiar-test and 512
unseen-test streams: 2,560 trials, 7.734 GiB of raw feature arrays. Capture uses
CUDA/BF16 and stores FP32 arrays; ridge fitting uses FP64. Separate probes use
nested training budgets of 128, 256, 512 and 1,024. Each budget fits its own
training-only feature scaling and selects its own penalty on full validation.

**Macro coordinate R²** averages `1 − SSE_j/TSS_j` across the eight task-vector
coordinates, where SSE is prediction squared error and TSS is the coordinate's
variation around its evaluation-set mean. Any constant evaluation coordinate
makes this metric undefined. These single-world scores have no across-world
uncertainty interval.

**Solver convergence** requires the largest relative normal-equation residual
across output coordinates to be at most `1e-6`. The ridge objective is mean
squared vector error plus `lambda * ||W||²`, with an unpenalized intercept.

**Exposure coverage** counts preceding tasks containing each of the two target
constituents, sorts the counts, and bins them as 0, 1, 2 or 3+. A reported cell
requires at least 20 trials per world and at least four worlds common to every
supported cell in its panel. R² additionally requires every target coordinate
to vary within each included world/cell.

## Resource and numerical results

All 40 repeated solver comparisons, 420 learning-curve ridge fits and 84 additional
penalty-check fits converged. There are also 12 training-mean predictors.

| Measurement | Result on one GH200 120GB |
| --- | ---: |
| Capture 2,560 trials | 112.95 s |
| Fit/select 420 ridge candidates and 12 means, including monitoring | 134.86 s |
| 40 direct/CG comparison solves | 94.17 s |
| 84 additional validation-only penalty candidates | 52.68 s |
| Largest measured Python-process host memory | 6.58 GiB |
| Largest measured ridge GPU allocation | 2.03 GiB |

The direct/CG comparison uses the final-answer representation, all 1,024 training
streams, five penalties from `1e-5` to `10`, and two repetitions per solver/penalty.
Median solve times below pool those ten measurements per family/method; the first
MLP direct call includes initialization and is slower than subsequent calls.

| Representation | Direct median | Streamed CG median | CG iterations |
| --- | ---: | ---: | ---: |
| All MLP hidden activations | 0.0168 s | 0.499 s | 21–125 |
| All fast-weight states | 0.659 s | 7.85 s | 33–37 |

The maximum direct/CG validation-prediction RMSE difference is `9.1e-7`; their
maximum objective difference is `6.3e-12`. Raising the explicit dense-memory
allowance from 1 to 4 GiB permits the full-state direct solve. Its estimated
working set is 2.055 GiB. Direct solving is therefore preferable at this budget.

## Learning curves and penalty range

At the final-answer capture, validation R² follows:

| Training streams | Trained state | Untrained state | Trained MLP | Untrained MLP |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.7204 | 0.5476 | −0.1324 | −0.1202 |
| 256 | 0.8566 | 0.7006 | −0.0324 | −0.0404 |
| 512 | 0.9196 | 0.7931 | 0.0405 | −0.0002 |
| 1,024 | 0.9525 | 0.8464 | 0.0829 | 0.0420 |

The original grid was `{1e-5, 1e-3, 0.1, 1, 10}`. All trained MLP probes selected
its upper edge. A validation-only check at 1,024 streams added `1e-7`, `100`,
`1,000` and `10,000`, reusing the same captures and target permutations. The trained
MLP optimum remained 10 and was bracketed by worse scores at 100. Stronger
penalties improved several untrained and shuffled controls, so the development
grid includes 100, 1,000 and 10,000. Reducing the state penalty from `1e-5` to
`1e-7` improved validation R² by less than `4.2e-9`; retain `1e-5` as the floor.

The original-grid held-out check at final answer gives state R² of **0.9740** on
familiar supports and **0.9455** on unseen supports, versus **0.9048/0.8394** for
the untrained state control. This is strong within-world decodability in this
calibration world. The substantial untrained signal makes the trained-control
difference important. These measurements neither establish an across-world
effect nor identify the model's causal algorithm. Larger training budgets have
not been exhausted, particularly for weak representation families.

## Exposure-cell coverage

A separate CPU audit generated exact target/history metadata for all eight planned
development worlds, all three target positions and both test populations. It
used nested held-out budgets of 512, 1,024 and 2,048. No model predictions were
used to choose the budget. D=2 avoids generating unused observations; task
latents, targets and exposures were checked against D=32 for the first stream
of each world/position/population, and their RNG streams are independent of D.

| Trials per world/position/test population | Middle-position unseen R² panel | All possible exposure cells reported? |
| --- | --- | --- |
| 512 | Only three common worlds; entire panel withheld | No |
| 1,024 | Four common worlds; seven of eight cells | No |
| 2,048 | Seven common worlds; all eight cells | Yes, in every audited panel |

At 2,048 trials, all panels use seven or eight common worlds. The minimum trial
and world thresholds remain unchanged. The saved W&B coverage figure shows every
position/population. Fresh confirmation worlds can have different coverage;
unsupported cells must still be withheld under the frozen rules.

## Development configuration and dispatch

`configs/probe/task_vector.yaml` specifies 1,024/512/2,048/2,048 streams, the eight
penalties above, and `solver.max_working_mb: 4096`. All other axes and controls
remain as defined in the experimental design. Eight worlds × three positions
give 135,168 independent captured trials, approximately **420.75 GiB of raw
features**, and 99,072 ridge candidates for all configured views, plus 72 means.
Fitted probe weights, metadata and plots require additional scratch space.

For the first full development dispatch, use at most eight GPU workers, four CPUs
per worker and **64 GiB host RAM per node**, with four workers per node. Initial
wall-time bounds of 30 minutes for capture, 60 minutes for fitting and 30 minutes
for evaluation allow headroom for shared storage and centralized reporting.
These are scheduling limits, not measured full-study completion estimates.
The three jobs remain sequential and share one W&B run. Confirmation follows
protocol freeze after reviewing the development study.

<details><summary>Provenance and operational checks</summary>

- Code commit for the GPU measurements: `163464b`.
- [Shared W&B run](https://wandb.ai/jublett-university-of-oxford/iccl-analysis/runs/1e55a7ce).
- Scratch root:
  `/scratch/u6oz/jharvey.u6oz/iccl-analysis/task-vector/scale-calibration-20261022-163464b`.
- Capture/solver job: `7642179`; completed in 3 min 43 s with one GPU, four CPUs,
  32 GiB RAM and a 20-minute limit.
- Fit/evaluation job: `7642236`; same allocation and limit. All numerical stages
  and W&B evaluation completed, but the temporary timing wrapper then failed to
  serialize the returned `Path`. Original fresh-evaluation timing is unavailable.
  The wrapper was corrected without changing pipeline code or numerical artifacts.
- Penalty/resume/publication job: `7642386`; one GPU, four CPUs, 32 GiB RAM,
  five-minute limit; completed in 1 min 37 s. Evaluation resume verified saved
  results and retained the same result identity; its 19.48 s is a resume/report
  time, not fresh evaluation.
- Peak GPU concurrency was one; all workloads ran in compute allocations.
  Login-node use was limited to Git, metadata, transfers and scheduling.
- All jobs have ended. Total allocated GPU time was 8 min 19 s, including the
  bookkeeping-failed job. The calibration scratch directory occupies about 11 GiB.
- The revised configuration resolves through Hydra and passes validation.
  Ten focused pipeline/sampling/reporting/dispatch tests passed; two CUDA tests
  were deselected locally. GPU solver checks are recorded separately above.
- Captures, weights, job logs and caches remain on scratch. Only small numerical
  summaries and figures were copied to the local ignored calibration folder.
- W&B calibration artifact contains solver diagnostics, coverage audits/figure,
  penalty checks, configuration and temporary drivers. The original timing
  driver is preserved alongside its serialization correction.
- Dataset: `277a385ecb786fc62357f6ea56fb30bcf619e3d8952efdaf68e992ad9a8c0415`.
- Probes: `9ecfc4712989b667a9395a8e379658dd05fe80bdacf1ae966ee6e5829fa21e74`.
- Results: `21813c067b7ad7025b4631d7d724bcdd2eca15218ba17086d658e4c7b7ca5f44`.

</details>
