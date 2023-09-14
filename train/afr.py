import torch
from tqdm import tqdm  

import wandb
wandb.login()

class Trainer:
    def __init__(self, model):
        self.model = model

    def first_stage_training(self, model, num_epochs, train_loader, loss_function, optimizer, device, config = None):
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
                wandb.log({"loss": loss})


    def second_stage_training(self, model, num_epochs, train_loader, loss_function, optimizer, weights, scheduler, device, config = None):
        with wandb.init(config = config):
            config = wandb.config

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

                wandb.log({"weighted_loss": weighted_loss})

                scheduler.step(weighted_loss)


    def compute_afr_weights(self, model, data, gamma, device):
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

            weights = (-gamma * p_true).exp()
            n_classes = torch.unique(y).numel()

            # class balancing
            class_count = []
            for y in range(n_classes):
                class_count.append((class_label == y).sum())

            for y in range(1, n_classes):
                weights[class_label == y] *= class_count[0] / class_count[y]
                
            weights /= weights.sum()
          
        return weights