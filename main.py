import hydra
import logging
from omegaconf import OmegaConf

import torch
import torch.nn as nn  
import torch.optim as optim  
import torchvision.transforms as transforms 
from torchvision.models import resnet50, ResNet50_Weights

from train import train_model
from models import CNN, LinearClassifier
from visualization import show_image 
from data_processing import (get_train_data, 
                             get_test_data, 
                             load_waterbirds_images_data, 
                             load_resnet50_representation_data)
from eval import (print_accuracy_for_loaders, 
                  print_group_accuracies_for_waterbirds_images, 
                  print_group_accuracies_for_resnet50_representation)


log = logging.getLogger(__name__)


@hydra.main(config_path = "conf", config_name = "config")
def main(cfg):
      log.info(OmegaConf.to_yaml(cfg))
      device = torch.device("cuda" if torch. cuda.is_available() else "cpu")

      model_name = cfg.model.name
      dataset_name = cfg.dataset.name

      if dataset_name == "waterbirds_images":
            train_dataset = get_train_data() 
            test_dataset = get_test_data()
            train_loader, test_loader = load_waterbirds_images_data(cfg.model.params.batch_size)

            show_image(train_dataset)

            if model_name == "resnet50":
                  model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1).to(device)
            elif model_name == "cnn":
                  model = CNN().to(device)
            else:
                  log.info(f"Combination not implemented: \n  dataset = {dataset_name}, \n  model = {model_name}")
                  return
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr = cfg.model.params.learning_rate, weight_decay = 1e-5)

            train_model(model, cfg.model.params.num_epochs, train_loader, criterion, optimizer, device)

            print_accuracy_for_loaders(train_loader, test_loader, model, device)
            print_group_accuracies_for_waterbirds_images(test_dataset, model, device)

      elif dataset_name == "resnet50_representation":
            train_path = cfg.dataset.paths.train_path
            test_path = cfg.dataset.paths.test_path
            train_loader, test_loader, train_dataset, test_dataset = load_resnet50_representation_data(train_path, test_path, cfg.model.params.batch_size)

            if model_name == "LinearClassifier":
                  model = LinearClassifier().to(device)
            else:
                  log.info(f"Combination not implemented: \n  dataset = {dataset_name}, \n  model = {model_name}")
                  return
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr = cfg.model.params.learning_rate, weight_decay = 1e-5)

            train_model(model, cfg.model.params.num_epochs, train_loader, criterion, optimizer, device)

            print_accuracy_for_loaders(train_loader, test_loader, model, device)
            print_group_accuracies_for_resnet50_representation(test_dataset, model, device)

 
if __name__ == "__main__":
      main()