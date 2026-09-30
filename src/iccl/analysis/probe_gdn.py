"""Matched GDN acquisition performance on the probe's exact worlds and fresh inputs."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from iccl.analysis.capture import CaptureEpisodes, collate_capture
from iccl.analysis.gdn_inference import continue_gdn
from iccl.analysis.probe_functional import functional_errors, functional_inputs, functional_targets
from iccl.checkpoints import (
    checkpoint_model_config,
    checkpoint_model_digest,
    resolve_checkpoint_path,
)
from iccl.data.sequences import TOKEN_X
from iccl.models.model import model_from_config
from iccl.models.ops import Backend, resolve_backend
from iccl.training.trainer import resolve_autocast_dtype

GDN_FUNCTIONAL_PROTOCOL = "after-task-demonstrations-independent-queries-v1"


class GDNFunctionalEvaluator:
    """Frozen source model, with independent queries after each task's 32 demonstrations.

    The probe sees the final episode state and oracle task coefficients; this
    baseline measures acquisition at each task's end. It is behavioral context,
    not a ceiling on information linearly accessible in the final matrices.
    """

    def __init__(
        self,
        manifest: dict[str, Any],
        split: str,
        settings: DictConfig,
        device: torch.device,
    ) -> None:
        identity = manifest["identity"]
        provenance = manifest["source_provenance"]
        reference = settings.checkpoint or provenance["checkpoint_reference"]
        path, _ = resolve_checkpoint_path(str(reference))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        digest = checkpoint_model_digest(checkpoint)
        if digest != identity["source_model_digest"]:
            raise ValueError("GDN evaluation checkpoint differs from the captured source model")
        architecture = OmegaConf.create(checkpoint_model_config(checkpoint))
        self.backend: Backend = resolve_backend(cast(Backend, settings.backend), device)
        architecture.model.backend = self.backend
        self.model = model_from_config(architecture).to(device).eval().requires_grad_(False)
        self.model.load_state_dict(checkpoint["model"])
        self.device = device
        self.dtype = resolve_autocast_dtype(settings.precision, device)
        self.batch_size = int(settings.batch_size)
        self.query_batch_size = int(settings.query_batch_size)
        self.tasks = int(identity["task_count"])
        data = OmegaConf.create(identity["data"])
        assert isinstance(data, DictConfig)
        self.episodes = CaptureEpisodes(
            data,
            int(identity["split_seeds"][split]),
            0,
            int(manifest["requested_counts"][split]),
        )
        self.metadata = {
            "protocol": GDN_FUNCTIONAL_PROTOCOL,
            "checkpoint_reference": str(reference),
            "checkpoint_path": str(Path(path).resolve()),
            "source_model_digest": digest,
            "source_step": int(checkpoint["step"]),
            "backend": self.backend,
            "precision": str(self.dtype or torch.float32),
            "device": str(device),
            "batch_size": self.batch_size,
            "query_batch_size": self.query_batch_size,
            "demonstrations_per_task": 32,
            "query_context": "original episode prefix through this task's final y-token",
            "queries_independent": True,
            "query_labels_revealed": False,
            "oracle_task_coefficients": False,
            "torch": str(torch.__version__),
        }

    @torch.inference_mode()
    def score(
        self, batch: dict[str, np.ndarray], *, inputs_per_task: int, seed: int
    ) -> dict[str, np.ndarray]:
        buffers: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(batch["episode_index"]), self.batch_size):
            selected = slice(start, start + self.batch_size)
            indices = batch["episode_index"][selected]
            generated = collate_capture([self.episodes[int(index)] for index in indices])
            for key in ("world_modules", "world_biases", "world_readout", "latents"):
                if not np.array_equal(generated[key], batch[key][selected]):
                    raise ValueError(f"regenerated GDN episodes differ from captured {key}")
            inputs = np.stack(
                [
                    functional_inputs(int(index), self.tasks, inputs_per_task, seed)
                    for index in indices
                ]
            )
            predictions = self.predict(generated, inputs)
            errors = functional_errors(predictions, functional_targets(generated, inputs))
            for key, value in errors.items():
                buffers.setdefault(key, []).append(value)
        return {key: np.concatenate(value) for key, value in buffers.items()}

    @torch.inference_mode()
    def predict(self, episodes: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
        """Fresh queries branch from unchanged task-end caches; original context continues."""
        batch, tasks, queries, width = inputs.shape
        if tasks != self.tasks or width != 16 or batch != len(episodes["episode_index"]):
            raise ValueError("fresh queries must match the episode/task population and input width")
        tokens = torch.from_numpy(episodes["tokens"]).to(self.device)
        types = torch.from_numpy(episodes["token_type"]).to(self.device)
        predictions = np.empty_like(inputs)
        cache = None
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
            for task in range(tasks):
                start, stop = task * 65, (task + 1) * 65
                _, cache = continue_gdn(
                    self.model,
                    tokens[:, start:stop],
                    types[:, start:stop],
                    cache,
                    backend=self.backend,
                )
                query_tokens = torch.from_numpy(inputs[:, task].reshape(-1, 1, 16)).to(self.device)
                outputs = []
                for offset in range(0, batch * queries, self.query_batch_size):
                    stop_query = min(offset + self.query_batch_size, batch * queries)
                    rows = torch.arange(offset, stop_query, device=self.device) // queries
                    values = query_tokens[offset:stop_query]
                    output, _ = continue_gdn(
                        self.model,
                        values,
                        torch.full(values.shape[:2], TOKEN_X, dtype=torch.long, device=self.device),
                        cache.select(rows),
                        backend=self.backend,
                    )
                    outputs.append(output[:, 0].float().cpu().numpy())
                predictions[:, task] = np.concatenate(outputs).reshape(batch, queries, 16)
        return predictions
