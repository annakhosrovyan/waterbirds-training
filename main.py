import hydra
import logging
from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch. nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 

from eval import (algorithm_performance,
                  standard_model_performance,
                  print_group_accuracies)

load_dotenv('.env')
log = logging.getLogger(__name__)


def configure_experiment(cfg):
      algorithm_name = cfg.algorithm.name
      model_name = cfg.model.name
      datamodule_name = cfg.datamodule.name

      return algorithm_name, model_name, datamodule_name


def configure_device(cfg):
      device = torch.device(cfg.device.name if torch.cuda.is_available() else "cpu")
    
      return device


def validate_data_formats(cfg):
      algorithm_data_format = set(cfg.algorithm.data_format)
      model_data_format = set(cfg.model.data_format)
      datamodule_format = set(cfg.datamodule.data_format)

      if not algorithm_data_format.intersection(model_data_format, datamodule_format):
            raise NotImplementedError("Incompatible data formats in algorithm, model, and datamodule configurations.")


@hydra.main(version_base = None, config_path = "conf", config_name = "config")
def main(cfg):

      log.info(OmegaConf.to_yaml(cfg))
      device = configure_device(cfg)

      algorithm_name, model_name, datamodule_name = configure_experiment(cfg)

      validate_data_formats(cfg)

# --------------------Create a neural network model --------------------

      # model = instantiate(cfg.model.network_config, 
      #                     optimizer_config = cfg.optimizer.optim, 
      #                     loss_config = cfg.loss.loss_function).to(device)

      model = instantiate(cfg.model.network_config,
                            loss_config = cfg.loss,
                            ).to(device)


      
# -----------------------Initialize optimizer and loss function -----------------------

      optimizer, scheduler = model.configure_optimizers()
      # optimizer = instantiate(cfg.optimizer, params = model.parameters())
     
      # loss_function = model.configure_loss()
      loss_function = instantiate(cfg.loss)

# --------------------Train the model and evaluate performance--------------------

      trainer = instantiate(cfg.algorithm.trainer, model, cfg.datamodule)

      train_loader, new_train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = trainer.prep_data(cfg.datamodule.batch_size, cfg.algorithm.params.weight, device)

      trainer.standard_training(model, cfg.algorithm.params.num_epochs, train_loader, test_loader, loss_function, optimizer, device)
      standard_model_performance(train_loader, test_loader, test_dataset, model, datamodule_name, device)

      trainer.second_stage_training(model, cfg.algorithm.params.num_epochs_for_final_model, new_train_loader, test_loader, loss_function, optimizer, cfg.algorithm.params.weight, device)
      
      log.info("\nValidation Group Accuracies\n")
      print_group_accuracies(val_dataset, model, datamodule_name, device)
      
      algorithm_performance(train_loader, test_loader, test_dataset, model, datamodule_name, device)


if __name__ == "__main__":
      main()
      