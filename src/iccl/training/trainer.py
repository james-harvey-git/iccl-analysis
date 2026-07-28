"""Meta-training loop: optimization, checkpointing, and W&B logging.

Trains the sequence model on the on-the-fly ICCL stream with a loss at every
demonstration position.
"""
