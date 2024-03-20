import torch
import torchmetrics
import pytorch_lightning as pl
import logging

from hydra.utils import instantiate

log = logging.getLogger(__name__)

class ERM(pl.LightningModule):
    def __init__(self, 
                 num_classes,
                 network, 
                 loss_fn, 
                 optimizer_config, 
                 scheduler_config,
                 *args, **kwargs):
        
        super().__init__()

        self.network = network
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.accuracy = torchmetrics.Accuracy(task='binary', 
                                              num_classes=num_classes)

        self._reset_counters()



    def forward(self, x):
        return self.network(x)

    def _step(self, batch, batch_idx):
        img, labels, backgrounds = batch
        scores = self.network(img)
        loss = self.loss_fn(scores, labels)
        
        return loss, scores, labels, backgrounds

    def training_step(self, train_batch, batch_idx):
        loss, scores, labels, backgrounds = self._step(train_batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, labels)
        self.log_dict({'train_loss': loss,
                       'train_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, scores, labels, backgrounds = self._step(val_batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, labels)
        self.log_dict({'val_loss': loss,
                       'val_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)

        self._check_group_accuracy(preds, labels, backgrounds)

        return loss
    
    def on_validation_epoch_end(self):
        self._log_val_acc()        
        self._reset_counters()

    def test_step(self, test_batch, batch_idx):
        loss, scores, labels, backgrounds = self._step(test_batch, batch_idx)
        preds = torch.argmax(scores, dim=1)

        accuracy = self.accuracy(preds, labels)
        self.log_dict({'test_loss': loss,
                       'test_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)
        
        self._check_group_accuracy(preds, labels, backgrounds)

        return loss
    
    def on_test_epoch_end(self):
        self._log_test_acc()
        self._reset_counters()
    
    def configure_optimizers(self):
        optimizer_target = self.optimizer_config.pop('target')
        optimizer = instantiate(self.optimizer_config, params = self.parameters(), _target_ = optimizer_target)

        monitor = self.scheduler_config.monitor
        del self.scheduler_config.monitor

        scheduler_target = self.scheduler_config.pop('target')
        scheduler = instantiate(self.scheduler_config, optimizer = optimizer, _target_ = scheduler_target)
        
        return {'optimizer': optimizer, 
                'lr_scheduler': scheduler, 
                'monitor': monitor}   
    
   
    def _reset_counters(self):
        self._group_total = torch.zeros(4)
        self._group_correct = torch.zeros(4)
        self._total = torch.tensor(0)
        self._correct = torch.tensor(0)

    def _check_group_accuracy(self, preds, labels, backgrounds):
        for idx, (i, j) in enumerate(zip(labels, backgrounds)):
            self._total += 1
            if i == 0 and j == 0:
                self._group_total[0] += 1
                if preds[idx] == i:
                    self._group_correct[0] += 1
                    self._correct += 1
            elif i == 0 and j == 1:
                self._group_total[1] += 1
                if preds[idx] == i:
                    self._group_correct[1] += 1
                    self._correct += 1
            elif i == 1 and j == 0:
                self._group_total[2] += 1
                if preds[idx] == i:
                    self._group_correct[2] += 1
                    self._correct += 1
            else:
                self._group_total[3] += 1
                if preds[idx] == i:
                    self._group_correct[3] += 1
                    self._correct += 1

    def _log_val_acc(self):
        self.log('val_accuracy', self._correct / self._total * 100)
        self.log('val_acc_landbird_land', self._group_correct[0] / self._group_total[0] * 100)
        self.log('val_acc_landbird_water', self._group_correct[1] / self._group_total[1] * 100)
        self.log('val_acc_waterbird_land', self._group_correct[2] / self._group_total[2] * 100)
        self.log('val_acc_waterbird_water', self._group_correct[3] / self._group_total[3] * 100)

    def _log_test_acc(self):
        self.log('test_accuracy', self._correct / self._total * 100)
        self.log('test_acc_landbird_land', self._group_correct[0] / self._group_total[0] * 100)
        self.log('test_acc_landbird_water', self._group_correct[1] / self._group_total[1] * 100)
        self.log('test_acc_waterbird_land', self._group_correct[2] / self._group_total[2] * 100)
        self.log('test_acc_waterbird_water', self._group_correct[3] / self._group_total[3] * 100)
