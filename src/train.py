import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utils import PrintingCallback
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(log_model=False, project='waterbirds_training')


def train(cfg: DictConfig) -> None:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "flexible" else cfg.device)

    datamodule = instantiate(cfg.datamodule.data_config)
    network = instantiate(cfg.networks)
    
    first_stage_model = instantiate(cfg.first_stage_model,
                                    network=network,
                                    loss_fn=cfg.loss,
                                    optimizer_config=cfg.optimizer.first_stage,
                                    scheduler_config=cfg.scheduler.first_stage
                                    ).to(device)
    
    first_stage_trainer = pl.Trainer(max_epochs=cfg.first_stage_model.num_epochs,
                                    accelerator=device,
                                    deterministic=True,
                                    callbacks=PrintingCallback(),
                                    # logger=wandb_logger
                                    )
    first_stage_trainer.fit(model=first_stage_model, datamodule=datamodule)
    first_stage_trainer.validate(model=first_stage_model, datamodule=datamodule)
    torch.save(first_stage_model.state_dict(), cfg.first_stage_model.weights_path)
    first_stage_trainer.test(model=first_stage_model, datamodule=datamodule)


    if datamodule.training_type != 'standard':
        datamodule.change_to_2nd_stage(model=first_stage_model, gamma=cfg.second_stage_model.gamma)
        network.freeze_layers(cfg.second_stage_model.freeze_option)

        second_stage_model = instantiate(cfg.second_stage_model,
                                        loss_fn=cfg.loss,
                                        network=network,
                                        optimizer_config=cfg.optimizer.second_stage,
                                        scheduler_config=cfg.scheduler.second_stage
                                        ).to(device)

        second_stage_trainer = pl.Trainer(max_epochs=cfg.second_stage_model.num_epochs, 
                                        accelerator=device,
                                        deterministic=True,
                                        callbacks=PrintingCallback(),
                                        logger=wandb_logger)
        second_stage_trainer.fit(model=second_stage_model, datamodule=datamodule)
        torch.save(second_stage_model.state_dict(), cfg.second_stage_model.weights_path)
        second_stage_trainer.validate(model=second_stage_model, datamodule=datamodule)
        second_stage_trainer.test(model=second_stage_model, datamodule=datamodule)
