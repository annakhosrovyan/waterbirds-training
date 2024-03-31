import torch
import torch.nn as nn  
import torchmetrics
import pytorch_lightning as pl

from hydra.utils import instantiate


class AFR(pl.LightningModule):
    def __init__(self, 
                 num_classes, 
                 reg_AFR,
                 network,
                 optimizer_config, 
                 scheduler_config,
                 freeze_option,
                 *args, **kwargs):
        
        super().__init__()

        self.network = network
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.reg_AFR = reg_AFR
        self.freeze_option = freeze_option
        self._reset_counters()
        self._initial_params = self.get_params()
        self.accuracy = torchmetrics.Accuracy(task = 'binary', num_classes = num_classes)
        self.save_hyperparameters()            

    def get_params(self):
        last_layer_params = []
        penultimate_layer_params = []
        last_two_layers_params = []
        for name, param in self.network.named_parameters():
            if self.network._penultimate_layer in name:
                penultimate_layer_params.append(param.data.clone().to("cuda"))
                last_two_layers_params.append(param.data.clone().to("cuda"))
            if self.network._last_layer in name:
                last_layer_params.append(param.data.clone().to("cuda"))
                last_two_layers_params.append(param.data.clone().to("cuda"))

        if self.freeze_option == 'unfreeze_penultimate':
            return penultimate_layer_params
        if self.freeze_option == 'unfreeze_last':
            return last_layer_params
        if self.freeze_option == 'unfreeze_both':
            return last_two_layers_params


    def forward(self, x):      
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        img, labels, weights = batch
        scores = self.network(img)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, labels)
        loss = self.loss_afr(scores, labels, weights, self.reg_AFR, self._initial_params)
        self.log_dict({'train_loss': loss,
                       'train_accuracy': accuracy * 100},
                       on_step = False,
                       on_epoch = True,
                       prog_bar = False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels, backgrounds  = batch

        scores = self.network(data)
        loss = self.loss_afr(scores, labels, torch.ones_like(labels), self.reg_AFR, self._initial_params)
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
        print("val_group_acc% ", self._group_correct / self._group_total * 100)
        self._log_val_acc()        
        self._reset_counters()

    def test_step(self, batch, batch_idx):
        data, labels, backgrounds = batch
        scores = self.network(data)
        loss = self.loss_afr(scores, labels, torch.ones_like(labels), self.reg_AFR, self._initial_params)
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

    def loss_afr(self, preds, labels, weights, reg_AFR, initial_params):
        pre_loss = nn.CrossEntropyLoss(reduction="none")
        weighted_afr_loss = torch.sum(weights * pre_loss(preds, labels))
        params = self.get_params()
        diff_params_list = [reg_AFR * ((param - init_param)) for param, init_param in
                             zip(params, initial_params)]
    
        l2_norm_squared = sum(torch.norm(param.view(-1), p=2) ** 2 for param in diff_params_list) 

        return torch.tensor(reg_AFR * l2_norm_squared, requires_grad=True) + weighted_afr_loss


    def configure_optimizers(self):
        optimizer_target = self.optimizer_config.pop('target')
        optimizer = instantiate(self.optimizer_config, params=self.parameters(), _target_=optimizer_target)
        
        return {'optimizer': optimizer}

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
        self.log('val_wga', torch.min(self._group_correct / self._group_total * 100))
        self.log('val_acc_landbird_land', self._group_correct[0] / self._group_total[0] * 100)
        self.log('val_acc_landbird_water', self._group_correct[1] / self._group_total[1] * 100)
        self.log('val_acc_waterbird_land', self._group_correct[2] / self._group_total[2] * 100)
        self.log('val_acc_waterbird_water', self._group_correct[3] / self._group_total[3] * 100)

    def _log_test_acc(self):
        self.log('test_accuracy', self._correct / self._total * 100)
        self.log('test_wga', torch.min(self._group_correct / self._group_total * 100))
        self.log('test_acc_landbird_land', self._group_correct[0] / self._group_total[0] * 100)
        self.log('test_acc_landbird_water', self._group_correct[1] / self._group_total[1] * 100)
        self.log('test_acc_waterbird_land', self._group_correct[2] / self._group_total[2] * 100)
        self.log('test_acc_waterbird_water', self._group_correct[3] / self._group_total[3] * 100)
