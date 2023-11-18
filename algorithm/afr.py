import torch
import logging
import pytorch_lightning as pl

from tqdm import tqdm  
from algorithm.erm import ERM
from torch.utils.data import DataLoader
from callbacks import PrintingCallback
from eval import print_group_accuracies

log = logging.getLogger(__name__)


class AFR(ERM):
    def __init__(self, 
                 model, 
                 data_module,
                 *args, **kwargs):
        
        super().__init__(model, data_module)
        

    def train(self):
        first_stage_data_loader, second_stage_data_loader, *_ = self.prepare_two_stage_data()

        ERM.train(self)

        self.freeze_all_except_last()

        trainer = pl.Trainer(max_epochs = self._cfg.algorithm.num_epochs_for_final_model,
                             callbacks= PrintingCallback())

        trainer.fit(model = self.model, train_dataloaders = second_stage_data_loader)
        trainer.validate(dataloaders = self.data_module.val_dataloader())
        trainer.test(dataloaders = self.data_module.test_dataloader())

        *_, test_dataset = self.data_module.setup()
        print_group_accuracies(test_dataset, self.model, 
                               self._cfg.datamodule.name)


    def freeze_all_except_last(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc2.parameters():
            param.requires_grad = True


    def compute_afr_weights(self, data, gamma):
        with torch.no_grad():
            erm_logits = []
            class_label = []
            for index, (x, y, _) in enumerate(tqdm(data)):
                x = x.to(device = self._device)
                y = y.to(device = self._device)
                probs = self.model(x.unsqueeze(0))
                erm_logits.append(probs)
                class_label.append(y)

            class_label = torch.cat([tensor.view(-1) for tensor in class_label], 
                                    dim = 0).to(device = self._device)
            erm_logits = torch.cat([tensor.view(-1, 2) for tensor in erm_logits], dim = 0)


            p = erm_logits.softmax(-1)
            y_onehot = torch.zeros_like(erm_logits).scatter_(-1, 
                                        class_label.unsqueeze(-1), 1).to(device = self._device)
            p_true = (p * y_onehot).sum(-1)

            weights = (-gamma * p_true).exp() 
            n_classes = torch.unique(class_label).numel()

            # class balancing
            class_count = []
            for y in range(n_classes):
                class_count.append((class_label == y).sum())
            
            for y in range(1, n_classes):
                weights[class_label == y] *= class_count[0] / class_count[y]
                
            # weights /= weights.sum()

        return weights


    def prepare_two_stage_data(self):
        train_dataset, *_ = self.data_module.setup()

        first_stage_data_size = int(len(train_dataset) * 0.8)
        second_stage_data_size = len(train_dataset) - first_stage_data_size
       
        first_stage_dataset, second_stage_dataset = torch.utils.data.random_split(train_dataset, 
                                                        [first_stage_data_size, second_stage_data_size])
        
        first_stage_data_loader = DataLoader(first_stage_dataset, 
                                             self.data_module.batch_size, 
                                             shuffle = True)

        second_stage_data_loader = DataLoader(second_stage_dataset, 
                                              self.data_module.batch_size, 
                                              shuffle = True)

        return (first_stage_data_loader, 
                second_stage_data_loader, 
                first_stage_dataset, 
                second_stage_dataset)
