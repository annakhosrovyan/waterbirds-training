import torch
from tqdm import tqdm  
from hydra.utils import instantiate
from torch.utils.data import DataLoader
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

        erm_data_size = int(len(train_dataset) * 0.8)
        rw_data_size = len(train_dataset) - erm_data_size
       
        erm_data, rw_data = torch.utils.data.random_split(train_dataset, [erm_data_size, rw_data_size])
        erm_data_loader = DataLoader(erm_data, batch_size, shuffle = True)
        weights = self.compute_afr_weights(self.model, erm_data, weight, device)
        rw_data_loader = DataLoader(rw_data, batch_size, shuffle = True)

        return erm_data_loader, rw_data_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

    def second_stage_training(self, model, num_epochs, train_loader, test_loader, loss_function, optimizer, weights, device, config = None):
    # def second_stage_training(self, model, num_epochs, train_loader, test_loader, loss_function, optimizer, weights, scheduler, device, config = None):
        with wandb.init(config = config):
            config = wandb.config

            for param in model.parameters():
                param.requires_grad= False

            for param in model.fc2.parameters():
                param.requires_grad = True

            for epoch in range(num_epochs):
                for batch_idx, (data, targets, _) in enumerate(tqdm(train_loader)):
                    data = data.to(device = device)
                    targets = targets.to(device = device)

                    scores = model(data)
                    loss = loss_function(scores, targets)

                    weighted_loss = torch.sum(weights * loss)

                    optimizer.zero_grad()
                    weighted_loss.backward()
        
                    optimizer.step()

                accuracy = check_accuracy(test_loader, model, device)
                wandb.log({"accuracy": accuracy})

                # scheduler.step(weighted_loss)


    def compute_afr_weights(self, model, data, weight, device):
        with torch.no_grad():
            erm_logits = []
            class_label = []
            for index, (x, y, _) in enumerate(tqdm(data)):
                x = x.to(device = device)
                y = y.to(device = device)
                probs = model(x.unsqueeze(0))
                erm_logits.append(probs)
                class_label.append(y)
            
            class_label = torch.tensor(class_label).to(device = device)
            erm_logits = torch.cat(erm_logits, dim=0).to(device = device)
            p = erm_logits.softmax(-1)
            y_onehot = torch.zeros_like(erm_logits).scatter_(-1, class_label.unsqueeze(-1), 1).to(device = device)
            p_true = (p * y_onehot).sum(-1)

            weights = (-weight * p_true).exp()
            n_classes = torch.unique(y).numel()

            # class balancing
            class_count = []
            for y in range(n_classes):
                class_count.append((class_label == y).sum())

            for y in range(1, n_classes):
                weights[class_label == y] *= class_count[0] / class_count[y]
                
            weights /= weights.sum()
          
        return weights