import torch
import torch.nn as nn  
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from hydra.utils import instantiate
from src.model.transformer_architecture import Transformer
from torch.optim.lr_scheduler import StepLR


class Polymorfer(pl.LightningModule):
    def __init__(self, 
                 in_features, 
                 num_classes, 
                 optimizer_config, 
                 scheduler_config, 
                 *args, **kwargs):
        
        super().__init__()

        self.tranformer = Transformer(embed_dim=in_features,
                                    prototype_dim=100,
                                    seq_len = 32,
                                    num_blocks=1,
                                    expansion_factor=1,
                                    dropout=0.2)

        self.fc = nn.Linear(100, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
    
        self.accuracy = torchmetrics.Accuracy(task='binary', 
                                              num_classes=num_classes)
    
    def forward(self, x):
        c = self.tranformer(x, x)
        c = self.fc(c)
        
        return c

    def step(self, batch, batch_idx):
        x, y, _ = batch
        scores = self(x)
        loss = self.loss_fn(scores, y)
        
        return loss, scores, y

    def training_step(self, train_batch, batch_idx):
        loss, scores, y = self.step(train_batch, batch_idx)
        _, pred = scores.max(1)
        accuracy = self.accuracy(pred, y)
        self.log_dict({'train_loss': loss,
                       'accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)

        return {'loss':loss, 
                'scores': scores, 
                'y': y
                }

    def validation_step(self, val_batch, batch_idx):
        loss, scores, y = self.step(val_batch, batch_idx)
        _, pred = scores.max(1)
        accuracy = self.accuracy(pred, y)
        self.log_dict({'val_loss': loss,
                       'accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)

        return {'loss':loss, 
                'scores': scores, 
                'y': y
                }

    def test_step(self, test_batch, batch_idx):
        loss, scores, y = self.step(test_batch, batch_idx)
        _, pred = scores.max(1)
        accuracy = self.accuracy(pred, y)
        self.log_dict({'test_loss': loss,
                       'accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)

        return {'loss':loss, 
                'scores': scores, 
                'y': y
                }

    def configure_optimizers(self):
    # Note: Planning to switch to Hydra soon for configuration management.    
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        monitor = 'val_loss'
        
        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': monitor}   
