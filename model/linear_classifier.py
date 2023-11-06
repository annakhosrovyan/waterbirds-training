import torch.nn as nn  

from hydra.utils import instantiate

  
class Model(nn.Module):
    def __init__(self, in_features, num_classes, loss, optimizer_config, scheduler_config, *args, **kwargs):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
        self.loss = loss
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


    def train_step(self, train_batch):
        x, y, _ = train_batch
        logits = self.forward(x)
        loss = self.loss_function(x, logits)

        return loss
    

    def configure_optimizers(self):
        optimizer_target = self.optimizer_config.pop('target')
        optimizer = instantiate(self.optimizer_config, params = self.parameters(), _target_ = optimizer_target)

        scheduler_target = self.scheduler_config.pop('target')
        scheduler = instantiate(self.scheduler_config, optimizer = optimizer, _target_ = scheduler_target)

        return optimizer, scheduler   
     
