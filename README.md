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
uv run python scripts/main.py
```
