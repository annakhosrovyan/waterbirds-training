import os
import torch
import pytorch_lightning as pl
import numpy as np

from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utils import PrintingCallback
from src.utils import print_group_accuracies
from pytorch_lightning.loggers import WandbLogger

from src.model.polymorfer import Polymorfer 

wandb_logger = WandbLogger(log_model="all")

def train(cfg: DictConfig) -> None:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "flexible" else cfg.device)

    datamodule = instantiate(cfg.datamodule.data_config)
    
    first_model = instantiate(cfg.first_model,
                              loss_fn=cfg.loss,
                              optimizer_config=cfg.optimizer,
                              scheduler_config=cfg.scheduler
                              ).to(device)
    
    first_trainer = pl.Trainer(max_epochs=cfg.first_model.num_epochs,
                               accelerator=device,
                               callbacks=PrintingCallback(),
                               logger=wandb_logger)
    first_trainer.fit(model=first_model, datamodule=datamodule)

    torch.save(first_model.state_dict(), cfg.first_model.weights_path)
    first_trainer.test(model=first_model, datamodule=datamodule)

    *_, test_dataset = datamodule.setup()
    print_group_accuracies(test_dataset, first_model,
                           cfg.datamodule.name)
    
    if datamodule.training_type != 'standard':
    
    # Note: Planning to switch to Hydra soon for configuration management.
        second_model = Polymorfer(in_features=768, 
                    num_classes=2, 
                    optimizer_config=cfg.optimizer, 
                    scheduler_config=cfg.scheduler)

        second_trainer = pl.Trainer(max_epochs=cfg.second_model.num_epochs, 
                                    accelerator=device,
                                    callbacks=PrintingCallback(),
                                    logger=wandb_logger)
        second_trainer.fit(model=second_model, datamodule=datamodule)
        torch.save(second_model.state_dict(), cfg.second_model.weights_path)
        second_trainer.validate(model=second_model, datamodule=datamodule)
        second_trainer.test(model=second_model, datamodule=datamodule)

        # *_, test_dataset = datamodule.setup()
        # print_group_accuracies(test_dataset, second_model, cfg.datamodule.name)
