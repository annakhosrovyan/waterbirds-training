import torch
import wandb
import logging

from tqdm import tqdm  
from torch.utils.data import DataLoader
from train.standard import StandardTrainer
from eval import (algorithm_performance,
                  standard_model_performance,
                  check_group_accuracy,
                  print_group_accuracies)

wandb.login()
log = logging.getLogger(__name__)


class Trainer(StandardTrainer):
    def __init__(self, model, datamodule, device, cfg):
        super().__init__(model, datamodule, device, cfg)


    def second_stage_training(self, num_epochs, train_loader, test_loader, loss_function, optimizer, scheduler):
        with wandb.init():
            for epoch in range(num_epochs):
                for batch_idx, (data, targets, _) in enumerate(tqdm(train_loader)):
                    data = data.to(device = self.device)
                    targets = targets.to(device = self.device)

                    scores = self.model(data)
                    loss = loss_function(scores, targets)

                    optimizer.zero_grad()
                    loss.backward()
        
                    optimizer.step()
                
                *_, val_dataset, _ = self.datamodule.prepare_data()
                wandb.log({"Accuracy on waterbird_land":  
                           check_group_accuracy(val_dataset, self.model, 1, 0, self.cfg.datamodule.name, self.device), 
                           "Accuracy on landbird_water": 
                           check_group_accuracy(val_dataset, self.model, 0, 1, self.cfg.datamodule.name, self.device)})

                scheduler.step(loss)


    def construct_error_set(self, data): 
        error_set = []
        for index, (x, y, _) in enumerate(tqdm(data)):
            x = x.to(device = self.device)
            y = y.to(device = self.device)

            _, pred = self.model(x.unsqueeze(0)).max(1)
            
            if pred != y:
                error_set.append(index)
        
        return error_set


    def construct_upsampled_dataset(self, data, lambda_up):
        upsampled_dataset = []

        error_set = self.construct_error_set(data)

        for index, (x, y, c) in enumerate(tqdm(data)):
            if index in error_set:
                for i in range(lambda_up):
                    upsampled_dataset.append((x, y, c))
            else:
                upsampled_dataset.append((x, y, c))

        return upsampled_dataset
    

    def prepare_two_stage_data(self):
        train_loader, _, _, train_dataset, *_ = self.datamodule.prepare_data()
        first_stage_dataset, first_stage_data_loader = train_dataset, train_loader

        second_stage_dataset = self.construct_upsampled_dataset(train_dataset, self.cfg.algorithm.params.lambda_up)
        second_stage_data_loader = DataLoader(second_stage_dataset, self.datamodule.batch_size, shuffle = True)

        return (first_stage_data_loader, second_stage_data_loader, 
                first_stage_dataset, second_stage_dataset)


    def fit(self):
        optimizer, scheduler = self.model.configure_optimizers()
       
        *_, test_loader, _, _, test_dataset = self.datamodule.prepare_data()
        first_stage_data_loader, second_stage_data_loader, *_ = self.prepare_two_stage_data()

        self.standard_training(self.cfg.algorithm.params.num_epochs, 
                               first_stage_data_loader, test_loader, self.model.loss, optimizer)
        standard_model_performance(first_stage_data_loader, test_loader, test_dataset, 
                                   self.model, self.cfg.datamodule.name, self.device)
        
        self.second_stage_training(self.cfg.algorithm.params.num_epochs_for_final_model, 
                                   second_stage_data_loader, test_loader, self.model.loss, optimizer, scheduler)        
        algorithm_performance(second_stage_data_loader, test_loader, test_dataset, 
                              self.model, self.cfg.datamodule.name, self.device)

