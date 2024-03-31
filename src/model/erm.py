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
        self._reset_train_counters()
        self._reset_val_counters()
        self._reset_test_counters()
        self.save_hyperparameters()

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
        self.log_dict({'ERM_train_loss': loss,
                       'ERM_train_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)
        
        self._check_group_accuracy_train(preds, labels, backgrounds)
  
        return loss

    def on_train_epoch_end(self):
        self._log_train_acc()        
        self._reset_train_counters()

    def validation_step(self, val_batch, batch_idx):
        loss, scores, labels, backgrounds = self._step(val_batch, batch_idx)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, labels)
        self.log_dict({'ERM_val_loss': loss,
                       'ERM_val_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)

        self._check_group_accuracy_val(preds, labels, backgrounds)

        return loss
    
    def on_validation_epoch_end(self):
        self._log_val_acc()        
        self._reset_val_counters()

    def test_step(self, test_batch, batch_idx):
        loss, scores, labels, backgrounds = self._step(test_batch, batch_idx)
        preds = torch.argmax(scores, dim=1)

        accuracy = self.accuracy(preds, labels)
        self.log_dict({'ERM_test_loss': loss,
                       'ERM_test_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)
        
        self._check_group_accuracy_test(preds, labels, backgrounds)

        return loss
    
    def on_test_epoch_end(self):
        self._log_test_acc()
        self._reset_test_counters
    
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
    
    def _reset_train_counters(self):
        self._group_total_train = torch.zeros(4)
        self._group_correct_train = torch.zeros(4)
        self._total_train = torch.tensor(0)
        self._correct_train = torch.tensor(0)

    def _reset_val_counters(self):
        self._group_total_val = torch.zeros(4)
        self._group_correct_val = torch.zeros(4)
        self._total_val = torch.tensor(0)
        self._correct_val = torch.tensor(0)

    def _reset_test_counters(self):
        self._group_total_test = torch.zeros(4)
        self._group_correct_test = torch.zeros(4)
        self._total_test = torch.tensor(0)
        self._correct_test = torch.tensor(0)


    def _check_group_accuracy_train(self, preds, labels, backgrounds):
        for idx, (i, j) in enumerate(zip(labels, backgrounds)):
            self._total_train += 1
            if i == 0 and j == 0:
                self._group_total_train[0] += 1
                if preds[idx] == i:
                    self._group_correct_train[0] += 1
                    self._correct_train += 1
            elif i == 0 and j == 1:
                self._group_total_train[1] += 1
                if preds[idx] == i:
                    self._group_correct_train[1] += 1
                    self._correct_train += 1
            elif i == 1 and j == 0:
                self._group_total_train[2] += 1
                if preds[idx] == i:
                    self._group_correct_train[2] += 1
                    self._correct_train += 1
            else:
                self._group_total_train[3] += 1
                if preds[idx] == i:
                    self._group_correct_train[3] += 1
                    self._correct_train += 1
        
    def _check_group_accuracy_val(self, preds, labels, backgrounds):
        for idx, (i, j) in enumerate(zip(labels, backgrounds)):
            self._total_val += 1
            if i == 0 and j == 0:
                self._group_total_val[0] += 1
                if preds[idx] == i:
                    self._group_correct_val[0] += 1
                    self._correct_val += 1
            elif i == 0 and j == 1:
                self._group_total_val[1] += 1
                if preds[idx] == i:
                    self._group_correct_val[1] += 1
                    self._correct_val += 1
            elif i == 1 and j == 0:
                self._group_total_val[2] += 1
                if preds[idx] == i:
                    self._group_correct_val[2] += 1
                    self._correct_val += 1
            else:
                self._group_total_val[3] += 1
                if preds[idx] == i:
                    self._group_correct_val[3] += 1
                    self._correct_val += 1

    def _check_group_accuracy_test(self, preds, labels, backgrounds):
        for idx, (i, j) in enumerate(zip(labels, backgrounds)):
            self._total_test += 1
            if i == 0 and j == 0:
                self._group_total_test[0] += 1
                if preds[idx] == i:
                    self._group_correct_test[0] += 1
                    self._correct_test += 1
            elif i == 0 and j == 1:
                self._group_total_test[1] += 1
                if preds[idx] == i:
                    self._group_correct_test[1] += 1
                    self._correct_test += 1
            elif i == 1 and j == 0:
                self._group_total_test[2] += 1
                if preds[idx] == i:
                    self._group_correct_test[2] += 1
                    self._correct_test += 1
            else:
                self._group_total_test[3] += 1
                if preds[idx] == i:
                    self._group_correct_test[3] += 1
                    self._correct_test += 1

    def _log_train_acc(self):
        self.log('ERM_train_accuracy', self._correct_train / self._total_train * 100)
        self.log('ERM_train_wga', torch.min(self._group_correct_train / self._group_total_train * 100))
        self.log('ERM_train_acc_landbird_land', self._group_correct_train[0] / self._group_total_train[0] * 100)
        self.log('ERM_train_acc_landbird_water', self._group_correct_train[1] / self._group_total_train[1] * 100)
        self.log('ERM_train_acc_waterbird_land', self._group_correct_train[2] / self._group_total_train[2] * 100)
        self.log('ERM_train_acc_waterbird_water', self._group_correct_train[3] / self._group_total_train[3] * 100)

    def _log_val_acc(self):
        self.log('ERM_val_accuracy', self._correct_val / self._total_val * 100)
        self.log('ERM_val_wga', torch.min(self._group_correct_val / self._group_total_val * 100))
        self.log('ERM_val_acc_landbird_land', self._group_correct_val[0] / self._group_total_val[0] * 100)
        self.log('ERM_val_acc_landbird_water', self._group_correct_val[1] / self._group_total_val[1] * 100)
        self.log('ERM_val_acc_waterbird_land', self._group_correct_val[2] / self._group_total_val[2] * 100)
        self.log('ERM_val_acc_waterbird_water', self._group_correct_val[3] / self._group_total_val[3] * 100)

    def _log_test_acc(self):
        self.log('ERM_test_accuracy', self._correct_test / self._total_test * 100)
        self.log('ERM_test_wga', torch.min(self._group_correct_test / self._group_total_test * 100))
        self.log('ERM_test_acc_landbird_land', self._group_correct_test[0] / self._group_total_test[0] * 100)
        self.log('ERM_test_acc_landbird_water', self._group_correct_test[1] / self._group_total_test[1] * 100)
        self.log('ERM_test_acc_waterbird_land', self._group_correct_test[2] / self._group_total_test[2] * 100)
        self.log('ERM_test_acc_waterbird_water', self._group_correct_test[3] / self._group_total_test[3] * 100)
