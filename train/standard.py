from tqdm import tqdm
from eval import check_accuracy

import wandb
wandb.login()

class StandardTrainer:
    def __init__(self):
        pass

    def standard_training(self, model, num_epochs, train_loader, test_loader, loss_function, optimizer, device, config = None):
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