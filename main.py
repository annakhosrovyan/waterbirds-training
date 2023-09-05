import hydra
import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch. nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 

from train import Trainer
from visualization import show_image 
from eval import (jtt_performance,
                  standard_model_performance,
                  print_group_accuracies)
from data_processing import WaterbirdsDatasetLoader, RepresentationDatasetLoader


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

      intersection = algorithm_data_format.intersection(model_data_format, dataset_format)
      if not intersection:
            raise NotImplementedError

# -----------------------Initialize the loss function-----------------------
   
      loss_function = instantiate(cfg.loss.loss_function)

# --------------------Load data based on the selected dataset--------------------

      representation_datasets_names = ["resnet50_representation", "dino_v2_representation", "regnet_representation"]

      if dataset_name == "waterbirds":
            waterbirds_loader = WaterbirdsDatasetLoader("C:/Users/User/Desktop/Datasets/waterbirds_v1.0", "waterbirds", download=True)
            train_dataset = waterbirds_loader.get_train_data() 
            test_dataset = waterbirds_loader.get_test_data()
            train_loader, test_loader = waterbirds_loader.load_data(cfg.model.params.batch_size)
            show_image(train_dataset)
            
      elif dataset_name in representation_datasets_names:
            train_path = cfg.dataset.paths.train_path
            test_path = cfg.dataset.paths.test_path
            val_path = cfg.dataset.paths.val_path
            resnet_loader = RepresentationDatasetLoader(train_path, test_path, val_path, cfg.model.params.batch_size)
            train_loader, test_loader, train_dataset, test_dataset, val_dataset = resnet_loader.load_data()

      else: 
            raise NotImplementedError

# --------------------Create a neural network model based on the chosen 'model_name'--------------------

      if model_name == "resnet50":
            model = instantiate(cfg.model.resnet50_config).to(device)
      elif model_name == "cnn":
            model = instantiate(cfg.model.cnn_config, cfg.model.params.in_channel, cfg.model.params.num_classes).to(device)
      elif model_name == "linear_classifier":
            model = instantiate(cfg.model.linear_config, cfg.dataset.in_features, cfg.model.params.num_classes).to(device)
      else: 
            raise NotImplementedError
      
# -----------------------Initialize the optimizer-----------------------

      optimizer = instantiate(cfg.optimizer.optim, params = model.parameters())

# -------------------Initialize the Learning Rate Scheduler-------------------

      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)

# --------------------Train the model and evaluate its performance--------------------

      trainer = Trainer(model)
      trainer.train_model(model, cfg.model.params.num_epochs, train_loader, loss_function, optimizer, scheduler, device)
      standard_model_performance(train_loader, test_loader, test_dataset, model, dataset_name, device)

# --------------------Apply the 'jtt' algorithm--------------------

      if algorithm_name == "jtt":
            upsampled_dataset = trainer.construct_upsampled_dataset(model, train_dataset, cfg.algorithm.params.lambda_up, device)
            new_train_loader = DataLoader(upsampled_dataset, cfg.model.params.batch_size, shuffle = True)
 
            trainer.train_model(model, cfg.algorithm.params.num_epochs_for_final_model, new_train_loader, loss_function, optimizer, scheduler, device)

            log.info("\nValidation Group Accuracies\n")
            print_group_accuracies(val_dataset, model, dataset_name, device)
            jtt_performance(train_loader, test_loader, test_dataset, model, dataset_name, device)
 
 
if __name__ == "__main__":
      main()