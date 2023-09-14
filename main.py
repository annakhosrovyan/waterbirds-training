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

from visualization import show_image 
from eval import (algorithm_performance,
                  standard_model_performance,
                  print_group_accuracies)

load_dotenv('.env')
log = logging.getLogger(__name__)


@hydra.main(version_base = None, config_path = "conf", config_name = "config")
def main(cfg):

      log.info(OmegaConf.to_yaml(cfg))
      device = torch.device("cuda" if torch. cuda.is_available() else "cpu")

      algorithm_name = cfg.algorithm.name
      model_name = cfg.model.name
      dataset_name = cfg.dataset.name

# --------------------Raise NotImplementedError for unsupported algorithm-dataset-model scenarios--------------------

      algorithm_data_format = set(cfg.algorithm.data_format)
      model_data_format = set(cfg.model.data_format)
      dataset_format = set(cfg.dataset.data_format)

      if not algorithm_data_format.intersection(model_data_format, dataset_format):
            raise NotImplementedError

# -----------------------Initialize the loss function-----------------------
   
      loss_function = instantiate(cfg.loss.loss_function)

# --------------------Load data--------------------

      if cfg.dataset.data_format not in ["vector", "image"]:
            raise NotImplementedError

      datamodule = instantiate(cfg.dataset)
      train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = datamodule.prepare_data()


# --------------------Create a neural network model based on the chosen 'model_name'--------------------

      if model_name == "resnet50":
            model = instantiate(cfg.model.resnet50_config).to(device)
      elif model_name == "cnn":
            model = instantiate(cfg.model, cfg.model.params.in_channel, 
                                cfg.model.params.num_classes, loss_function).to(device)
      elif model_name == "linear_classifier":
            model = instantiate(cfg.model, cfg.dataset.in_features, 
                                cfg.model.params.num_classes, loss_function).to(device)
      else: 
            raise NotImplementedError

# -----------------------Initialize the optimizer-----------------------

      optimizer = instantiate(cfg.optimizer.optim, params = model.parameters())

# -------------------Initialize the Learning Rate Scheduler-------------------

      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, verbose=True)

# --------------------Train the model and evaluate performance--------------------

      trainer = instantiate(cfg.algorithm.trainer, model)

      if algorithm_name == "afr":
            erm_data_size = int(len(train_dataset) * 0.8)
            rw_data_size = len(train_dataset) - erm_data_size
            erm_data, rw_data = torch.utils.data.random_split(train_dataset, [erm_data_size, rw_data_size])
            erm_data_loader = DataLoader(erm_data, cfg.dataset.batch_size, shuffle = True)
            rw_data_loader = DataLoader(rw_data, cfg.dataset.batch_size, shuffle = True)
            trainer.first_stage_training(model, cfg.model.params.num_epochs, erm_data_loader, loss_function, optimizer, device)
            standard_model_performance(erm_data_loader, test_loader, test_dataset, model, dataset_name, device)

            weights = trainer.compute_afr_weights(model, erm_data, cfg.algorithm.params.gamma, device)
            rw_data_loader = DataLoader(rw_data, cfg.dataset.batch_size, shuffle = True)
            trainer.second_stage_training(model, cfg.algorithm.params.num_epochs_for_final_model, rw_data_loader,
                                           loss_function, optimizer,  weights, scheduler, device)

            log.info("\nValidation Group Accuracies\n")
            print_group_accuracies(val_dataset, model, dataset_name, device)
            algorithm_performance(rw_data_loader, test_loader, test_dataset, model, dataset_name, device)

            return


      
      trainer.first_stage_training(model, cfg.model.params.num_epochs, train_loader, loss_function, optimizer, device)
      standard_model_performance(train_loader, test_loader, test_dataset, model, dataset_name, device)


# --------------------Apply the 'jtt' algorithm--------------------

      if algorithm_name == "jtt":
            upsampled_dataset = trainer.construct_upsampled_dataset(model, train_dataset, cfg.algorithm.params.lambda_up, device)
            new_train_loader = DataLoader(upsampled_dataset, cfg.dataset.batch_size, shuffle = True)
 
            trainer.second_stage_training(model, cfg.algorithm.params.num_epochs_for_final_model, new_train_loader, loss_function, optimizer, scheduler, device)

            log.info("\nValidation Group Accuracies\n")
            print_group_accuracies(val_dataset, model, dataset_name, device)
            algorithm_performance(train_loader, test_loader, test_dataset, model, dataset_name, device)
 
 
if __name__ == "__main__":
      main()
      