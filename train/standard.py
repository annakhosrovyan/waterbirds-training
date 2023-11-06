import wandb

from tqdm import tqdm
from eval import check_accuracy

wandb.login()

class StandardTrainer:
    def __init__(self, model, datamodule, device, cfg):
        self.model = model
        self.datamodule = datamodule
        self.device = device
        self.cfg = cfg
 

    def standard_training(self, num_epochs, train_loader, test_loader, loss_function, optimizer):
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
                accuracy = check_accuracy(test_loader, self.model, self.device)
                wandb.log({"accuracy": accuracy})


    def fit(self):
        optimizer, _ = self.model.configure_optimizers()
        train_loader, _, test_loader, *_ = self.datamodule.prepare_data()
        self.standard_training(self.cfg.algorithm.params.num_epochs, train_loader, test_loader, self.model.loss, optimizer)