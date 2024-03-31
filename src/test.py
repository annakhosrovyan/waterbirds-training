import torch
import pytorch_lightning as pl

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.model import ERM, AFR

def test(cfg: DictConfig) -> None:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "flexible" else cfg.device)

    datamodule = instantiate(cfg.datamodule.data_config)
    network = instantiate(cfg.networks)

    first_stage_model = ERM.load_from_checkpoint(checkpoint_path=cfg.first_stage_model.checkpoints.load.path).to(device)

    first_stage_model.eval()

    first_stage_trainer = pl.Trainer(accelerator=device)
    first_stage_trainer.validate(model=first_stage_model, datamodule=datamodule)
    first_stage_trainer.test(model=first_stage_model, datamodule=datamodule)

    if datamodule.training_type != 'standard':
        datamodule.change_to_2nd_stage(model=first_stage_model, gamma=cfg.second_stage_model.gamma)
        network.freeze_layers(cfg.second_stage_model.freeze_option)

        second_stage_model = AFR.load_from_checkpoint(checkpoint_path=cfg.second_stage_model.checkpoints.load.path).to(device)
        second_stage_model.eval()

        second_stage_trainer = pl.Trainer(accelerator=device)
        second_stage_trainer.validate(model=second_stage_model, datamodule=datamodule)
        second_stage_trainer.test(model=second_stage_model, datamodule=datamodule)