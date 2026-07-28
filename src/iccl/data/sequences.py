"""ICCL sequence construction on top of the HyperTeacher task family.

Turns sampled tasks into a single in-context continual-learning sequence: tasks
are presented one after another within one context window, each as a run of
(input, output) demonstrations. The model predicts the output at every
demonstration position, giving a dense learning signal. Primitives are
re-instantiated per sequence, so the in-context algorithm must learn them from
demonstrations rather than memorize them across sequences.
"""
