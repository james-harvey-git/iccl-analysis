"""Export of frozen evaluation and analysis sets.

Training data stays on-the-fly and effectively infinite; the sequences used for
reported evaluation numbers and for mechanistic interpretability are exported
once to disk as small artifacts, so they remain byte-identical regardless of
later changes to the sampler. Entry point: ``scripts/make_eval_sets.py``.
"""
