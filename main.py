import hydra
import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torchvision.models import resnet50, ResNet50_Weights

from train import train_model, construct_upsampled_dataset
from models import CNN, LinearClassifier
from visualization import show_image 
from data_processing import (get_train_data, 
                             get_test_data, 
                             load_waterbirds_images_data, 
                             load_resnet50_representation_data)
from eval import (jtt_performance,
                  standard_model_performance,
                  print_group_accuracies)


log = logging.getLogger(__name__)


@hydra.main(version_base = None, config_path = "conf", config_name = "config")
def main(cfg):
      log.info(OmegaConf.to_yaml(cfg))
      device = torch.device("cuda" if torch. cuda.is_available() else "cpu")

      algorithm_name = cfg.algorithm.name
      model_name = cfg.model.name
      dataset_name = cfg.dataset.name


# --------------------Raise NotImplementedError for unsupported algorithm-dataset-model scenarios--------------------

      if algorithm_name == "jtt":
            if dataset_name != "resnet50_representation":
                  raise NotImplementedError

      if dataset_name == "waterbirds_images":
            if model_name == "linear_classifier":
                  raise NotImplementedError
            
      if dataset_name == "resnet50_representation":
            if model_name != "linear_classifier":
                  raise NotImplementedError


# -----------------------Initialize the loss function-----------------------

      criterion = instantiate(cfg.loss.loss_function)


# --------------------Load data based on the selected dataset--------------------

      if dataset_name == "waterbirds_images":
            train_dataset = get_train_data() 
            test_dataset = get_test_data()
            train_loader, test_loader = load_waterbirds_images_data(cfg.model.params.batch_size)
            show_image(train_dataset)
            
      elif dataset_name == "resnet50_representation":
            train_path = cfg.dataset.paths.train_path
            test_path = cfg.dataset.paths.test_path
            val_path = cfg.dataset.paths.val_path
            train_loader, test_loader, train_dataset, test_dataset, val_dataset = load_resnet50_representation_data(train_path, test_path, 
                                                                                    val_path,  cfg.model.params.batch_size)
      else: 
            raise NotImplementedError
      

# --------------------Create a neural network model based on the chosen 'model_name'--------------------

      if model_name == "resnet50":
            model = instantiate(cfg.model.resnet50_config).to(device)
      elif model_name == "cnn":
            model = CNN().to(device)
      elif model_name == "linear_classifier":
            model = LinearClassifier().to(device)
      else: 
            raise NotImplementedError
      

# -----------------------Initialize the optimizer-----------------------

      optimizer = instantiate(cfg.optimizer.optim, params = model.parameters())
      

# --------------------Train the model and evaluate its performance--------------------

      train_model(model, cfg.model.params.num_epochs, train_loader, criterion, optimizer, device)
      standard_model_performance(train_loader, test_loader, test_dataset, model, dataset_name, device)


# --------------------Apply the 'jtt' algorithm--------------------

      if algorithm_name == "jtt":
            upsampled_dataset = construct_upsampled_dataset(model, train_dataset, cfg.algorithm.params.lambda_up, device)
            new_train_loader = DataLoader(upsampled_dataset, cfg.model.params.batch_size, shuffle = True)
 
            train_model(model, cfg.algorithm.params.num_epochs_for_final_model, new_train_loader, criterion, optimizer, device)

            log.info("\nValidation Group Accuracies\n")
            print_group_accuracies(val_dataset, model, dataset_name, device)
            jtt_performance(train_loader, test_loader, test_dataset, model, dataset_name, device)

 
if __name__ == "__main__":
      main()

