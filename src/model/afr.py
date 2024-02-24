import torch
import torch.nn as nn  
import torchmetrics
import pytorch_lightning as pl
import copy

from hydra.utils import instantiate


class AFR(pl.LightningModule):
    def __init__(self, 
                 datamodule,
                 first_model,
                 num_classes, 
                 loss_fn, 
                 optimizer_config, 
                 scheduler_config,
                 gamma, 
                 *args, **kwargs):
        
        super().__init__()

        self.datamodule = datamodule
        self.fc1 = copy.deepcopy(first_model.fc1) 
        self.fc2 = copy.deepcopy(first_model.fc2) 
        
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.accuracy = torchmetrics.Accuracy(task = 'binary', 
                                              num_classes = num_classes)
        
        for param in self.parameters():
            param.requires_grad = False

        for param in self.fc2.parameters():
            param.requires_grad = True

        self.corrects = torch.zeros(4)
        self.totals = torch.zeros(4)
        self.worst_total = torch.tensor(0)
        self.worst_correct = torch.tensor(0)
        self.val_accuracy = 0
        self.test_accuracy = 0
        
        self.weights = self.datamodule.change_to_2nd_stage(model=first_model, gamma=gamma) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        data, labels, _ = batch
        batch_weights = self.weights[batch_idx * self.datamodule.batch_size:
                        (batch_idx + 1) * self.datamodule.batch_size].to("cuda") 

        preds = self(data)
        loss = self.loss_afr(preds, labels, batch_weights)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels, backgrounds = batch

        preds = self(data)
        loss = self.loss_afr(preds, labels, torch.ones_like(labels))
        preds = torch.argmax(preds, dim=1)
        accuracy_2 = (preds == labels).float().mean()
        
        self._calculate_accuracy(labels, backgrounds, preds)
        self._calculate_worst_accuracy()

        self.val_accuracy = (self.worst_correct.float() / self.worst_total.float() if self.worst_total.float() != 0 
                                else torch.tensor(0))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy_2, on_step=False, on_epoch=True, prog_bar=True)
       
        return loss
    
    def on_validation_epoch_end(self):
        self._reset_counters()
        self.log('val_accuracy_WGA', torch.tensor(self.val_accuracy), on_epoch=True, prog_bar=True)
        self.val_accuracy = 0

    def test_step(self, batch, batch_idx):
        data, labels, backgrounds = batch
        preds = self(data)
        loss = self.loss_afr(preds, labels, torch.ones_like(labels))
        preds = torch.argmax(preds, dim=1)
        accuracy_2 = (preds == labels).float().mean()
        
        self._calculate_accuracy(labels, backgrounds, preds)
        self._calculate_worst_accuracy()

        self.test_accuracy = (self.worst_correct.float() / self.worst_total.float() if self.worst_total.float() != 0 
                                else torch.tensor(0))

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', accuracy_2, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_test_epoch_end(self):
        self._reset_counters()
        self.log('test_WGA_accuracy', torch.tensor(self.test_accuracy), on_epoch=True, prog_bar=True)
        self.val_accuracy = 0

    def loss_afr(self, preds, labels, weights):
        pre_loss = nn.CrossEntropyLoss(reduction="none")
        weighted_afr_loss = torch.sum(weights * pre_loss(preds, labels))
 
        return weighted_afr_loss
    
    def configure_optimizers(self):
        optimizer_target = self.optimizer_config.pop('target')
        optimizer = instantiate(self.optimizer_config, params=self.parameters(), _target_=optimizer_target)

        monitor = self.scheduler_config.monitor
        del self.scheduler_config.monitor

        scheduler_target = self.scheduler_config.pop('target')
        scheduler = instantiate(self.scheduler_config, optimizer=optimizer, _target_=scheduler_target)
        
        self.optimizer_config.update({'target': optimizer_target})  # for second stage
        self.scheduler_config.update({'monitor': monitor, 'target': scheduler_target})

        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': monitor}   

    def _reset_counters(self):
        self.corrects = torch.zeros(4)
        self.totals = torch.zeros(4)
        self.worst_total = torch.tensor(0)
        self.worst_correct = torch.tensor(0)

    def _get_index(self, i, j):
        combinations = {(0, 0): 0, 
                        (0, 1): 1, 
                        (1, 0): 2, 
                        (1, 1): 3}
        
        return combinations.get((i, j))
    
    def _calculate_worst_accuracy(self):
        result = [self.corrects[i] / self.totals[i] if self.totals[i] != 0 
                        else 0 for i in range(len(self.totals))]
        minn = min(result)
        index = result.index(minn)
        self.worst_total = self.totals[index]
        self.worst_correct = self.corrects[index]

    def _calculate_accuracy(self, labels, backgrounds, preds):
        for idx, (i, j) in enumerate(zip(labels, backgrounds)):
            k = self._get_index(i.item(), j.item())
            self.totals[k] += 1
            if preds[idx] == i:
                self.corrects[k] += 1