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
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.fc2.parameters():
                param.requires_grad = True

            *_, train_dataset, val_dataset, _ = self.datamodule.prepare_data()
            weights = self.compute_afr_weights(train_dataset, self.cfg.algorithm.params.gamma)
            weights_loader = DataLoader(weights, self.datamodule.batch_size, shuffle = True)
          
            for epoch in range(num_epochs):
                for weight, (batch_idx, (data, targets, _)) in zip(weights_loader, enumerate(tqdm(train_loader))):
                    data = data.to(device = self.device)
                    targets = targets.to(device = self.device)

                    scores = self.model(data)
                    loss = loss_function(scores, targets)                     
                    weighted_loss = torch.sum(weight * loss)
                    
                    optimizer.zero_grad()
                    weighted_loss.backward()
        
                    optimizer.step()
                    
                wandb.log({"Accuracy on waterbird_land":  
                           check_group_accuracy(val_dataset, self.model, 1, 0, self.cfg.datamodule.name, self.device), 
                           "Accuracy on landbird_water": 
                           check_group_accuracy(val_dataset, self.model, 0, 1, self.cfg.datamodule.name, self.device)})
    
                scheduler.step(weighted_loss)


    def compute_afr_weights(self, data, gamma):
        with torch.no_grad():
            erm_logits = []
            class_label = []
            for index, (x, y, _) in enumerate(tqdm(data)):
                x = x.to(device = self.device)
                y = y.to(device = self.device)
                probs = self.model(x.unsqueeze(0))
                erm_logits.append(probs)
                class_label.append(y)

            class_label = torch.cat([tensor.view(-1) for tensor in class_label], 
                                    dim = 0).to(device = self.device)
            erm_logits = torch.cat([tensor.view(-1, 2) for tensor in erm_logits], dim = 0)


            p = erm_logits.softmax(-1)
            y_onehot = torch.zeros_like(erm_logits).scatter_(-1, 
                                        class_label.unsqueeze(-1), 1).to(device = self.device)
            p_true = (p * y_onehot).sum(-1)

            weights = (-gamma * p_true).exp() 
            n_classes = torch.unique(class_label).numel()

            # class balancing
            class_count = []
            for y in range(n_classes):
                class_count.append((class_label == y).sum())
            
            for y in range(1, n_classes):
                weights[class_label == y] *= class_count[0] / class_count[y]
                
            # weights /= weights.sum()

        return weights



    def prepare_two_stage_data(self):
        *_, train_dataset, _, _ = self.datamodule.prepare_data()

        first_stage_data_size = int(len(train_dataset) * 0.8)
        second_stage_data_size = len(train_dataset) - first_stage_data_size
       
        first_stage_dataset, second_stage_dataset = torch.utils.data.random_split(train_dataset, 
                                                        [first_stage_data_size, second_stage_data_size])
        first_stage_data_loader = DataLoader(first_stage_dataset, self.datamodule.batch_size, shuffle = True)

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
