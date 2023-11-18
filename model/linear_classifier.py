import torch.nn as nn  
import torchmetrics
import pytorch_lightning as pl

from hydra.utils import instantiate

  
class Model(pl.LightningModule):
    def __init__(self, 
                 in_features, 
                 num_classes, 
                 loss_fn, 
                 optimizer_config, 
                 scheduler_config, 
                 *args, **kwargs):
        
        super().__init__()

        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.accuracy = torchmetrics.Accuracy(task = 'binary', 
                                              num_classes = num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


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
        optimizer_target = self.optimizer_config.pop('target')
        optimizer = instantiate(self.optimizer_config, params = self.parameters(), _target_ = optimizer_target)

        monitor = self.scheduler_config.monitor
        del self.scheduler_config.monitor

        scheduler_target = self.scheduler_config.pop('target')
        scheduler = instantiate(self.scheduler_config, optimizer = optimizer, _target_ = scheduler_target)
        
        self.optimizer_config.update({'target': optimizer_target})  # for second stage
        self.scheduler_config.update({'monitor': monitor, 'target': scheduler_target})

        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': monitor}   
