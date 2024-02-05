import os
import torch
import pytorch_lightning as pl

from hydra.utils import instantiate
from omegaconf import DictConfig


def test(cfg: DictConfig) -> None:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "flexible" else cfg.device)

    datamodule = instantiate(cfg.datamodule.data_config)
    datamodule.cfg = cfg

    first_model = instantiate(cfg.first_model,
                        loss_fn=cfg.loss,
                        optimizer_config=cfg.optimizer,
                        scheduler_config=cfg.scheduler
                        ).to(device)
    first_model.load_state_dict(torch.load(cfg.first_model.weights_path))
    first_model.eval()

    first_trainer = pl.Trainer(max_epochs=cfg.first_model.num_epochs, 
                               accelerator=device)
    first_trainer.test(model=first_model, datamodule=datamodule)

    if cfg.training_type.name != 'standard':

        datamodule.change_to_2nd_stage(model=first_model)

        second_model = instantiate(cfg.second_model,
                                first_model=first_model,
                                loss_fn=cfg.loss,
                                optimizer_config=cfg.optimizer,
                                scheduler_config=cfg.scheduler,
                                datamodule=datamodule
                                ).to(device)

        second_model.load_state_dict(torch.load(cfg.second_model.weights_path))
        second_model.eval()

        second_trainer = pl.Trainer(max_epochs=cfg.second_model.num_epochs, accelerator=device)
        second_trainer.test(model=second_model, datamodule=datamodule)
