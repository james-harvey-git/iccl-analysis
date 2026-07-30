import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from iccl.data.dataset import (
    SequenceDataset,
    collate_sequences,
    make_family,
    sequence_config_from,
)
from iccl.utils import resolve_device, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    print(OmegaConf.to_yaml(cfg))
    print(f"device: {device}")

    family = make_family(cfg.data)
    dataset = SequenceDataset(family, sequence_config_from(cfg.data), base_seed=cfg.seed)
    loader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, collate_fn=collate_sequences
    )
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    shapes = {k: tuple(v.shape) for k, v in batch.items()}
    x_positions = int(batch["loss_mask"].sum().item())
    print(f"batch shapes: {shapes}; x-positions with loss: {x_positions}")


if __name__ == "__main__":
    main()
