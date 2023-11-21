
import logging
import pytorch_lightning as pl

from tqdm import tqdm  
from algorithm.erm import ERM
from torch.utils.data import DataLoader
from callbacks import PrintingCallback
from eval import print_group_accuracies

log = logging.getLogger(__name__)


class JTT(ERM):
    def __init__(self, 
                 model, 
                 data_module,
                 *args, **kwargs):

        super().__init__(model, data_module)


    def train(self):

        first_stage_data_loader, second_stage_data_loader, *_ = self.prepare_two_stage_data()

        ERM.train(self)

        trainer = pl.Trainer(accelerator = self._device,
                             max_epochs = self._cfg.algorithm.num_epochs_for_final_model,
                             callbacks = PrintingCallback())

        trainer.fit(model = self.model, train_dataloaders = second_stage_data_loader)
        trainer.test(dataloaders = self.data_module.test_dataloader())

        *_, test_dataset = self.data_module.setup()
        print_group_accuracies(test_dataset, self.model, 
                               self._cfg.datamodule.name)


    def construct_error_set(self, data): 
        error_set = []
        for index, (x, y, _) in enumerate(tqdm(data)):
            x = x.to(device = self._device)
            y = y.to(device = self._device)

            _, pred = self.model(x.unsqueeze(0)).max(1)
            
            if pred != y:
                error_set.append(index)
        
        return error_set


    def construct_upsampled_dataset(self, data, lambda_up):
        upsampled_dataset = []

        error_set = self.construct_error_set(data)

        for index, (x, y, c) in enumerate(tqdm(data)):
            if index in error_set:
                for i in range(lambda_up):
                    upsampled_dataset.append((x, y, c))
            else:
                upsampled_dataset.append((x, y, c))

        return upsampled_dataset
    

    def prepare_two_stage_data(self):
        first_stage_dataset, *_ = self.data_module.setup()
        first_stage_data_loader = self.data_module.train_dataloader()

        second_stage_dataset = self.construct_upsampled_dataset(first_stage_dataset, 
                                                                self._cfg.algorithm.lambda_up)
        second_stage_data_loader = DataLoader(second_stage_dataset, 
                                              self.data_module.batch_size, 
                                              shuffle = True)

        return (first_stage_data_loader, 
                second_stage_data_loader, 
                first_stage_dataset, 
                second_stage_dataset)
    

