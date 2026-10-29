# Training performance and parity protocol

This study asks whether the token embedding and reduced padding can accelerate the
existing GDN training calculation without changing the data, objective, parameters,
checkpoint schema, or optimizer updates beyond floating-point accumulation error.

The production candidate is the per-type affine embedding in `model.py`. The indexed
implementation at main revision `4f49da4` is retained only as an experimental control
in `scripts/benchmark_training.py`. `group2` and `group4` are experimental wrappers:
they sort complete episodes within the original batch, trim each group's right
padding, and restore prediction order before the ordinary pooled masked loss. They
perform one backward pass and one optimizer update per original batch. They do not
change the sampler or split an episode's recurrent or convolution history. Padded
predictions are immaterial to the objective and may differ.

## Checks before timing and training comparisons

- Float64 CPU checks compare the affine mapping, input/parameter gradients, complete
  model predictions on real tokens, pooled loss, and every model parameter gradient.
- The ordinary golden-stream and checkpoint-resume regression tests must pass.
- CUDA checks run on identical batch tensors, with a SHA-256 digest recorded, both
  from initialization and a trained full-state checkpoint. Compare predictions,
  masked loss, every gradient tensor, and the subsequent AdamW parameter update.
- Checkpoint keys, parameter ordering and weight-decay exclusions must stay unchanged.
- Existing CUDA FLA/reference forward and backward parity tests must also pass on
  the actual Isambard environment before training results are accepted.

The diagnostic engineering tolerances are fixed before the GPU experiment:

| Quantity | FP32 relative RMS | BF16 relative RMS |
|---|---:|---:|
| Real-token predictions | 0.00005 | 0.005 |
| All parameter gradients together | 0.0001 | 0.01 |
| Any individual parameter gradient | 0.001 | 0.02 |
| AdamW parameter update together | 0.001 | 0.02 |
| Loss (relative absolute difference) | 0.00005 | 0.002 |

These are numerical consistency checks, not permission to change the algorithm.
Per-tensor maxima and absolute errors are retained. Repeated baseline evaluations
measure the nondeterministic reduction floor. Failed candidates are reported rather
than accepted by widening the limits after inspecting results.

## Timing

Use the configured batch size of 128 and the current variable-world data. Construct
exactly the worker-interleaved indices that the production loader would produce.
Every variant receives the same batches, initial parameters and optimizer state.
Warm every measured sequence shape, synchronize at timing-window boundaries, take
three timed passes, and repeat the baseline at the end. Record step times, losses,
peak allocated GPU memory, source revision, environment and batch digest. This
isolates device-resident training; actual `Trainer.fit` runs measure end-to-end time.

## Learning and continuation

Use the unmodified `Trainer.fit` with only the experimental model computation changed.
Compare baseline and selected candidate at matching optimizer steps, using three
fresh model-initialization seeds and the same indexed training data. Run a separate
paired continuation from the completed 3.5M-step full-state checkpoint, including
its optimizer and scheduler. Existing production training is not altered.

Record training objective, frozen validation error and the canonical capability
curves. Overlay conditions for each seed; distinguish optimizer-step progress from
within-episode capability curves. Keep complete numerical histories and final
checkpoints. Target 2,000 steps for each fresh run and 1,000 additional steps for
the continuation; the scheduler limit is an operational bound, not a scientific
convergence claim. At most six GPUs run these experiments concurrently.

Require finite training and final validation differences within 1% of the paired
baseline for every tested initialization and continuation. Inspect intermediate
curves and capability differences as well; matching the final scalar alone is not
sufficient. Report paired episode uncertainty when rescoring the final checkpoints.
These bounded experiments do not establish identical trajectories or equivalence
over a multi-million-step run.
