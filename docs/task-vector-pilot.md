# Task-vector pipeline development pilot

The purpose was to check the implementation and obtain a first resource estimate.
This pilot does not establish decoding performance across teacher worlds.

The source GDN was the step-2,900,000 checkpoint from run `0bdfkn9e`,
`weights-variable-surplus-s0-4-400k:v14`. One fixed world used M=8, two active
constituents, T=9, D=32 and target positions 1/5/9. Each position had 64 training,
32 validation, 32 familiar-test and 32 unseen-test streams: **480 trials total**.
All three capture points and all five concatenated representation families were
retained, with the trained/untrained, shuffled-target, observation and mean controls.
The two lambda candidates were 0.1 and 1.0.

## Observed cost

Isambard identified the accelerator as **NVIDIA GH200 120GB**. One GPU was used.
Capture used BF16 autocast with FP32 stored arrays; ridge solves used FP64.
Stage timings include initialization, I/O and W&B reporting. Evaluation also
includes local figure rendering and upload.

| Stage | Wall time |
| --- | ---: |
| Capture | 34.03 s |
| Fit and select 288 ridge candidates | 14.84 s |
| Held-out evaluation and reporting | 50.05 s |

Raw feature storage was **1.494 GiB**. Peak Python-process host memory was
**2.17 GiB**, and peak allocated CUDA memory was **226 MiB**. The job requested
32 GiB host memory and four CPUs; Slurm accounts 72 CPUs for the one-GPU
allocation. These are distinct from the process memory measurements.

The warmed measurements below use the final-answer representation at position 9,
64 training trials, lambda=1, one warmup and two measured repetitions. Values
are the median of those two repetitions.

| Input | Features | Direct solve | Streamed CG | CG iterations |
| --- | ---: | ---: | ---: | ---: |
| All MLP hidden activations | 4,096 | 2.64 ms | 22.29 ms | 18 |
| All fast-weight states | 131,072 | 36.16 ms | 128.64 ms | 11 |

Both solvers converged to the configured relative residual tolerance of 1e−6.
The direct and CG objectives agreed to better than 1e−12 in these checks.
Small-n direct-solve timings do not predict the cost of larger-n streaming CG.
The 1,024/512/512/512 capture plan at this pilot was approximately 191 GiB at this
checkpoint's dimensions. The subsequent [scale calibration](task-vector-calibration.md)
sets the development budgets and measures representative larger training sets.

## Validation and provenance

Five CUDA checks passed before the pilot: terminal capture under FP32 and BF16,
CUDA direct/CG ridge against CPU, and existing complete-cache continuation checks
under both precisions. The BF16 kernel comparison uses a BF16 reference so
precision drift is not conflated with backend correctness.

The separate stages resumed the same
[W&B run](https://wandb.ai/jublett-university-of-oxford/iccl-analysis/runs/0bd9cf9a),
with 48 media files including the live fit figure and world-averaged evaluation
figures. There were no failed fitting/evaluation units in that run.

The local suite passed **277 tests**, with lint, formatting and type checks clean.
A subsequent two-worker smoke run at commit `70d3b89` exercised the actual
submission helper, one visible GPU per worker, independent world ownership and
success-dependent stages. It used two worlds, position 1, D=2 and the smoke
32/12/16/12 split sizes, with W&B disabled to avoid another online pilot run.

| Dispatcher stage | Slurm job | Allocation wall time |
| --- | --- | ---: |
| Capture | 7641925 | 63 s |
| Fit | 7641927 | 31 s |
| Evaluate/report | 7641928 | 34 s |

Each job requested two GPUs, two CPUs per worker, 32 GiB per node and a five-minute
limit. All completed, and the stages did not overlap. Peak GPU concurrency across
the development checks was **two**. Eight-worker resource requests and duplicate
submission guards were also checked against a fake scheduler locally.

<details><summary>Exact execution record</summary>

- Commit: `fa835e8`.
- Slurm job: `7641737`, completed in **3 min 37 s**, including the CUDA gate.
- Allocation: one GPU, 32 GiB host RAM, 20-minute limit.
- Root on Isambard:
  `$SCRATCHDIR/iccl-analysis/task-vector/pilot-20261022-fa835e8`.
- Timing record:
  `development/logs/benchmark_probe/2026-10-22_20-08-27/benchmark.json`.
- Dataset identity:
  `44042b33e96baa6bcb31e76340c65b35def3aee1a94dd413c53275f0c8f401cf`.
- Probe identity:
  `fd72be336825e3684b864a9d7d412523d78fa16d9c1bd2f1b39642369a70ed25`.
- Result identity:
  `aecbeb736b16fd153e2e6d5897a044eacff2437ca2a4e6eca7ae8478773753cc`.

The implementation's numerical results, local report, figures and held-out
predictions remain under that root. The experiment configuration and source
checkpoint digest are saved in its manifests.

The two-worker check is under
`$SCRATCHDIR/iccl-analysis/task-vector/dispatch-check-20261022-70d3b89`.

</details>
