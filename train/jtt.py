import torch
from tqdm import tqdm  
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from train.standard import StandardTrainer
from eval import check_accuracy

import wandb
wandb.login()

class Trainer(StandardTrainer):
    def __init__(self, model, datamodule):
        super().__init__()
        self.model = model
        self.datamodule = instantiate(datamodule)

    def prep_data(self, batch_size, weight, device):
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = self.datamodule.prepare_data()
        upsampled_dataset = self.construct_upsampled_dataset(self.model, train_dataset, weight, device)
        new_train_loader = DataLoader(upsampled_dataset, batch_size, shuffle = True)

        return train_loader, new_train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

    def second_stage_training(self, model, num_epochs, train_loader, test_loader, loss_function, optimizer, weights, device, config = None):
    # def second_stage_training(self, model, num_epochs, train_loader, test_loader, loss_function, optimizer, weights, scheduler, device, config = None):
        with wandb.init(config = config):
            config = wandb.config

            for epoch in range(num_epochs):
                for batch_idx, (data, targets, _) in enumerate(tqdm(train_loader)):
                    data = data.to(device = device)
                    targets = targets.to(device = device)

                    scores = model(data)
                    loss = loss_function(scores, targets)

                    optimizer.zero_grad()
                    loss.backward()
        
                    optimizer.step()
                accuracy = check_accuracy(test_loader, model, device)
                wandb.log({"accuracy": accuracy})

                # scheduler.step(loss)


    def construct_error_set(self, model, data, device): 
        error_set = []
        for index, (x, y, _) in enumerate(tqdm(data)):
            x = x.to(device = device)
            y = y.to(device = device)

            _, pred = model(x.unsqueeze(0)).max(1)
            
            if pred != y:
                error_set.append(index)
        
        return error_set


    def construct_upsampled_dataset(self, model, data, weight, device):
        upsampled_dataset = []

        error_set = self.construct_error_set(model, data, device)

        for index, (x, y, c) in enumerate(tqdm(data)):
            if index in error_set:
                for i in range(weight):
                    upsampled_dataset.append((x, y, c))
            else:
                upsampled_dataset.append((x, y, c))

        return upsampled_dataset