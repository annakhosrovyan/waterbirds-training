import torch.nn as nn  
import torch.nn.functional as F

from hydra.utils import instantiate


class Model(nn.Module):
    def __init__(self, in_channels, num_classes, loss, optimizer_config, scheduler_config, *args, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(in_features = 200704, out_features = 2)

        self.loss = loss
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x
    

    def train_step(self, train_batch):
        x, y, _ = train_batch
        logits = self.forward(x)
        loss = self.loss(x, logits)
        
        return loss
    

    def configure_optimizers(self):
        optimizer_target = self.optimizer_config.pop('target')
        optimizer = instantiate(self.optimizer_config, params = self.parameters(), _target_ = optimizer_target)

        scheduler_target = self.scheduler_config.pop('target')
        scheduler = instantiate(self.scheduler_config, optimizer = optimizer, _target_ = scheduler_target)

        return optimizer, scheduler   
