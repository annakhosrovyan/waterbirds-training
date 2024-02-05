import os
import torch
import pytorch_lightning as pl

from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utils import PrintingCallback
from src.utils import print_group_accuracies
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(log_model="all")


def train(cfg: DictConfig) -> None:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "flexible" else cfg.device)

    datamodule = instantiate(cfg.datamodule.data_config)
    datamodule.cfg = cfg
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
    first_model = instantiate(cfg.first_model,
                              loss_fn=cfg.loss,
                              optimizer_config=cfg.optimizer,
                              scheduler_config=cfg.scheduler
                              ).to(device)
<<<<<<< Updated upstream

=======
    
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream

    second_model = instantiate(cfg.second_model,
                               loss_fn=cfg.loss,
                               optimizer_config=cfg.optimizer,
                               scheduler_config=cfg.scheduler,
                               datamodule=datamodule
                               ).to(device)
    second_model.cfg = cfg
=======
    
    if cfg.training_type.name != 'standard':
    
        second_model = instantiate(cfg.second_model,
                                first_model=first_model,
                                loss_fn=cfg.loss,
                                optimizer_config=cfg.optimizer,
                                scheduler_config=cfg.scheduler,
                                datamodule=datamodule
                                ).to(device)
>>>>>>> Stashed changes

        datamodule.change_to_2nd_stage(model=first_model)

<<<<<<< Updated upstream
    second_trainer = pl.Trainer(max_epochs=cfg.second_model.num_epochs, accelerator=device,
                                callbacks=PrintingCallback(),
                                logger=wandb_logger)
    second_trainer.fit(model=second_model, datamodule=datamodule)
    torch.save(second_model.state_dict(), os.path.join(weights_dir, 'second_model.pth'))
    second_trainer.validate(model=second_model, datamodule=datamodule)
    second_trainer.test(model=second_model, datamodule=datamodule)

    *_, test_dataset = datamodule.setup()
    print_group_accuracies(test_dataset, second_model, cfg.datamodule.name)
=======
        second_trainer = pl.Trainer(max_epochs=cfg.second_model.num_epochs, accelerator=device,
                                    callbacks=PrintingCallback(),
                                    logger=wandb_logger)
        second_trainer.fit(model=second_model, datamodule=datamodule)
        torch.save(second_model.state_dict(), cfg.second_model.weights_path)
        second_trainer.validate(model=second_model, datamodule=datamodule)
        second_trainer.test(model=second_model, datamodule=datamodule)

        *_, test_dataset = datamodule.setup()
        print_group_accuracies(test_dataset, second_model, cfg.datamodule.name)
>>>>>>> Stashed changes
