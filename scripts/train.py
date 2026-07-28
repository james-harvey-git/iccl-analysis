import hydra
from omegaconf import DictConfig, OmegaConf

from iccl.utils import resolve_device, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    print(OmegaConf.to_yaml(cfg))
    print(f"device: {device}")


if __name__ == "__main__":
    main()
