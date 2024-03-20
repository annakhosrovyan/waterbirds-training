import torch
import torch.nn as nn  
import torchmetrics
import pytorch_lightning as pl

from hydra.utils import instantiate


class AFR(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 loss_fn, 
                 reg_AFR,
                 network,
                 optimizer_config, 
                 scheduler_config,
                 *args, **kwargs):
        
        super().__init__()

        self.network = network
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.reg_AFR = reg_AFR

        self._reset_counters()

        self.init_last_layer = []
        for name, param in self.network.named_parameters():
            if 'fc' in name:
                self.init_last_layer.append(param.data.clone().to("cuda"))

        self.accuracy = torchmetrics.Accuracy(task = 'binary', 
                                              num_classes = num_classes)


    def forward(self, x):      
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        img, labels, weights = batch
        scores = self.network(img)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, labels)
        loss = self.loss_afr(scores, labels, weights, self.reg_AFR, self.init_last_layer)
        self.log_dict({'train_loss': loss,
                       'train_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels, backgrounds = batch

        scores = self.network(data)
        loss = self.loss_afr(scores, labels, torch.ones_like(labels), self.reg_AFR, self.init_last_layer)
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

    def test_step(self, batch, batch_idx):
        data, labels, backgrounds = batch
        scores = self.network(data)
        loss = self.loss_afr(scores, labels, torch.ones_like(labels), self.reg_AFR, self.init_last_layer)
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

    def loss_afr(self, preds, labels, weights, reg_AFR, initial_last_layer):
        pre_loss = nn.CrossEntropyLoss(reduction="none")
        weighted_afr_loss = torch.sum(weights * pre_loss(preds, labels))
        fc_params = []
        for name, param in self.network.named_parameters():
            if 'fc' in name:
                fc_params.append(param.data.clone().to("cuda"))
        diff_params_list = [init_param - fc_param for init_param, fc_param in
                             zip(initial_last_layer, fc_params)]
        l2_norm_squared = sum(torch.norm(param.view(-1), p=2) ** 2 for param in diff_params_list)

        return torch.tensor(reg_AFR * l2_norm_squared, requires_grad=True) + weighted_afr_loss


    def configure_optimizers(self):
        optimizer_target = self.optimizer_config.pop('target')
        optimizer = instantiate(self.optimizer_config, params=self.parameters(), _target_=optimizer_target)

        monitor = self.scheduler_config.monitor
        del self.scheduler_config.monitor

        scheduler_target = self.scheduler_config.pop('target')
        scheduler = instantiate(self.scheduler_config, optimizer=optimizer, _target_=scheduler_target)
        
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
