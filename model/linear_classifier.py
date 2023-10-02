import torch
import torch.nn as nn  
from hydra.utils import instantiate

  
class Model(nn.Module):
    def __init__(self, in_features, num_classes, loss_config, *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, num_classes)
        # self.loss_function = instantiate(loss_config)
        self.loss_config = loss_config
        # self.optimizer_config = optimizer_config

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    # def configure_loss(self):
    #     return instantiate(self.loss_config)

    # def train_step(self, train_batch):
    #     x, y, _ = train_batch
    #     logits = self.forward(x)
    #     loss = self.loss_function(x, logits)
    #     return loss
    
    # def configure_optimizers(self):
    #     optimizer = instantiate(self.optimizer_config, params = self.parameters())   
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, verbose=True)  # -> conf

    #     return optimizer, scheduler   
     
