import pytorch_lightning as pl

from pytorch_lightning import Trainer
from callbacks import PrintingCallback
from eval import print_group_accuracies


class ERM(pl.LightningModule):
    def __init__(self, 
                 model, 
                 data_module,
                 *args, **kwargs):
        
        super().__init__()

        self.model = model
        self.data_module = data_module

        self._device = None
        self._cfg = None


    @property
    def device(self):
       return self._device

    @device.setter
    def device(self, value):
        self._device = value

    @property
    def cfg(self):
        return self._cfg
    
    @cfg.setter
    def cfg(self, value):
        self._cfg = value

        
    def train(self):
        trainer = pl.Trainer(accelerator = self._device,
                             max_epochs = self._cfg.algorithm.num_epochs,
                             callbacks = PrintingCallback())

        trainer.fit(model = self.model, datamodule = self.data_module)
        trainer.test(dataloaders = self.data_module.test_dataloader())
        
        *_, test_dataset = self.data_module.setup()
        print_group_accuracies(test_dataset, self.model, 
                               self._cfg.datamodule.name)

