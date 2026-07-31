import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from iccl.training.trainer import Trainer
from iccl.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    print(OmegaConf.to_yaml(cfg))
    out_dir = HydraConfig.get().runtime.output_dir
    trainer = Trainer(cfg, out_dir=out_dir)
    num_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"device: {trainer.device}; model: {num_params:,} params; run dir: {out_dir}")
    trainer.fit()


if __name__ == "__main__":
    main()
