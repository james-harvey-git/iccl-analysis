from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from iccl.data.dataset import collate_sequences, sequence_rng, to_tensors
from iccl.data.sequences import (
    TOKEN_BOUNDARY,
    TOKEN_PAD,
    TOKEN_X,
    PhaseConfig,
    SequenceConfig,
    build_sequence,
)
from iccl.data.teacher import HyperTeacher, TeacherConfig
from iccl.models.model import GDNModel, TokenEmbedding, model_from_config

CONFIG_DIR = Path(__file__).parent.parent / "configs"

D_IN = 4


def small_model(**overrides: object) -> GDNModel:
    kwargs: dict = dict(
        d_in=D_IN,
        d_out=D_IN,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ffw=64,
        backend="reference",
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return GDNModel(**kwargs)


def small_batch(num_sequences: int = 2) -> dict[str, torch.Tensor]:
    family = HyperTeacher(
        TeacherConfig(
            input_dim=D_IN,
            output_dim=D_IN,
            hidden_dims=(D_IN,),
            use_bias=True,
            num_modules=8,
            scale=3.0,
            weighting="discrete",
        ),
        max_hotness=2,
    )
    cfg = SequenceConfig(
        phases=(PhaseConfig(num_tasks=8, hotness=(2, 2)),),
        demos_per_task=4,
        signal_boundaries=True,
        require_identifiable=True,
    )
    samples = [
        to_tensors(build_sequence(family, cfg, sequence_rng(0, i))) for i in range(num_sequences)
    ]
    return collate_sequences(samples)


def test_forward_shapes_on_real_batch() -> None:
    batch = small_batch()
    model = small_model()
    out = model(batch["tokens"], batch["token_type"])
    assert out.preds.shape == (*batch["tokens"].shape[:2], D_IN)
    assert out.states is None and out.hidden is None
    assert torch.isfinite(out.preds).all()


def test_full_model_causality() -> None:
    batch = small_batch()
    model = small_model()
    cut = batch["tokens"].shape[1] // 2
    perturbed = batch["tokens"].clone()
    perturbed[:, cut:] += 10.0
    with torch.no_grad():
        base = model(batch["tokens"], batch["token_type"]).preds
        shifted = model(perturbed, batch["token_type"]).preds
    torch.testing.assert_close(base[:, :cut], shifted[:, :cut])
    assert not torch.allclose(base[:, cut:], shifted[:, cut:])


def test_padding_does_not_affect_earlier_positions() -> None:
    batch = small_batch(num_sequences=1)
    model = small_model()
    seq_len = batch["tokens"].shape[1]
    pad = 7
    padded = {
        "tokens": torch.cat([batch["tokens"], torch.zeros(1, pad, D_IN)], dim=1),
        "token_type": torch.cat(
            [batch["token_type"], torch.full((1, pad), TOKEN_PAD, dtype=batch["token_type"].dtype)],
            dim=1,
        ),
    }
    with torch.no_grad():
        base = model(batch["tokens"], batch["token_type"]).preds
        with_pad = model(padded["tokens"], padded["token_type"]).preds
    torch.testing.assert_close(with_pad[:, :seq_len], base)


def test_capture_shapes_and_consistency() -> None:
    batch = small_batch()
    model = small_model()
    out = model(batch["tokens"], batch["token_type"], capture=True)
    assert out.states is not None and out.hidden is not None
    assert len(out.states) == len(out.hidden) == 2
    b, t = batch["tokens"].shape[:2]
    for states, hidden in zip(out.states, out.hidden, strict=True):
        assert states.shape == (b, t, 2, 32, 16)  # heads, d_v = 2 * d_k, d_k
        assert hidden.shape == (b, t, 32)
    # The last block's residual stream feeds the head directly.
    recomputed = model.head(model.final_norm(out.hidden[-1]))
    torch.testing.assert_close(recomputed, out.preds)


def test_capture_requires_reference_backend() -> None:
    batch = small_batch()
    model = small_model(backend="fla")
    with pytest.raises(ValueError, match="reference"):
        model(batch["tokens"], batch["token_type"], capture=True)


def test_state_dict_roundtrip_across_backend_settings() -> None:
    batch = small_batch()
    source = small_model(backend="auto")
    torch.manual_seed(1)
    target = small_model(backend="reference")
    target.load_state_dict(source.state_dict())
    with torch.no_grad():
        a = source(batch["tokens"], batch["token_type"], backend="reference").preds
        b = target(batch["tokens"], batch["token_type"]).preds
    torch.testing.assert_close(a, b)


def test_token_embedding_type_dependence() -> None:
    embed = TokenEmbedding(d_in=D_IN, d_model=8)
    with torch.no_grad():
        embed.bias.uniform_(-1.0, 1.0)
    tokens = torch.randn(1, 2, D_IN)
    tokens[0, 1] = tokens[0, 0]
    types = torch.tensor([[TOKEN_X, TOKEN_BOUNDARY]])
    h = embed(tokens, types)
    # Same content, different type: different embedding.
    assert not torch.allclose(h[0, 0], h[0, 1])
    # A content-free token embeds to exactly its per-type bias vector.
    zero = embed(torch.zeros(1, 1, D_IN), torch.tensor([[TOKEN_BOUNDARY]]))
    torch.testing.assert_close(zero[0, 0], embed.bias[TOKEN_BOUNDARY])


def test_model_from_config_builds_pilot_model() -> None:
    cfg = OmegaConf.create(
        {
            "data": OmegaConf.load(CONFIG_DIR / "data" / "hyperteacher.yaml"),
            "model": OmegaConf.load(CONFIG_DIR / "model" / "gdn.yaml"),
        }
    )
    model = model_from_config(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    assert 3e6 < num_params < 8e6
    tokens = torch.randn(1, 6, max(cfg.data.input_dim, cfg.data.output_dim))
    types = torch.tensor([[TOKEN_BOUNDARY, TOKEN_X, 1, TOKEN_X, 1, TOKEN_PAD]])
    out = model(tokens, types, backend="reference")
    assert out.preds.shape == (1, 6, cfg.data.output_dim)


def test_no_weight_decay_flags_present() -> None:
    model = small_model()
    flagged = {
        name
        for name, param in model.named_parameters()
        if getattr(param, "_no_weight_decay", False)
    }
    expected = {f"blocks.{i}.mixer.{p}" for i in range(2) for p in ("A_log", "dt_bias")}
    assert flagged == expected
