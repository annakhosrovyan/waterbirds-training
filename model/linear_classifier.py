import torch.nn as nn  
import torch.nn.functional as F
  
class Model(nn.Module):
    def __init__(self, in_features, num_classes, loss_function, *args, **kwargs):
        super(Model, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

        self.loss_function = loss_function

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def train_step(self, train_batch):
        x, y, _ = train_batch
        logits = self.forward(x)
        loss = self.loss_function(x, logits)
        return loss