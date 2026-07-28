import torch

from iccl.utils import resolve_device, seed_everything


def test_resolve_device_explicit() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_available_backend() -> None:
    device = resolve_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}


def test_seed_everything_makes_torch_deterministic() -> None:
    seed_everything(0)
    first = torch.randn(4)
    seed_everything(0)
    second = torch.randn(4)
    assert torch.equal(first, second)
