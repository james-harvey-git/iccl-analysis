"""HyperTeacher synthetic task, ported from smonsays/scale-compositionality.

Source: ``compscale/data/teacher.py`` (JAX/Flax). Primitives are the hidden-layer
modules of a one-hidden-layer teacher MLP shared across all tasks; a task is a
k-hot weighting (``num_hot`` of ``num_modules``) over those modules, applied to
the shared modules to produce the task's input-output function. Compositional
generalization is tested by holding out a subset of module combinations
(``task_support`` / ``frac_ood``) from the training task distribution.

Our port replaces JAX with PyTorch/NumPy and re-instantiates primitives per
sequence rather than fixing them for the whole dataset.
"""

class HyperTeacher:
    """Samples primitive modules and task functions for one family of tasks.

    Constructor parameters mirror ``configs/data/hyperteacher.yaml``.
    """

    def __init__(self) -> None:
        raise NotImplementedError("HyperTeacher port is the first implementation stage")
