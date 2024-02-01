import torch

from hydra.utils import instantiate
from omegaconf import DictConfig


def train(cfg: DictConfig):
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "flexible" else cfg.device)

    model = instantiate(cfg.model.network_config,
                        loss_fn=cfg.loss,
                        optimizer_config=cfg.optimizer,
                        scheduler_config=cfg.scheduler
                        ).to(device)

    data_module = instantiate(cfg.datamodule.data_config)

    algorithm = instantiate(cfg.algorithm,
                            data_module=data_module,
                            model=model
                            )
    algorithm.device = device
    algorithm.cfg = cfg

    algorithm.train()
