"""Activation and memory-state capture for mechanistic interpretability.

Runs the model with the reference backend (fused kernels hide their internals)
on frozen analysis sets, recording residual-stream activations and the per-step
fast-weight memory states S_t from each layer.
"""
