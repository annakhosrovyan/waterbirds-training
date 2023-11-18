import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datamodule.encoding import EncodingDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_path, 
                 test_path, 
                 val_path, 
                 batch_size, 
                 *args, **kwargs
                 ):
        
        super().__init__()

        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.batch_size = batch_size


    def get_data(self, path):
        data = np.load(path)
        x, y, c = data["embeddings"], data["labels"], data["domains"]
        
        return x.squeeze(), y.squeeze(), c.squeeze()


    def setup(self, stage = None):
        train_x, train_y, train_c = self.get_data(self.train_path)
        test_x, test_y, test_c = self.get_data(self.test_path)
        val_x, val_y, val_c = self.get_data(self.val_path)

        self.train_dataset = EncodingDataset(train_x, train_y, train_c)
        self.val_dataset = EncodingDataset(val_x, val_y, val_c)
        self.test_dataset = EncodingDataset(test_x, test_y, test_c)

        return self.train_dataset, self.val_dataset, self.test_dataset
    

    def train_dataloader(self):
        return DataLoader(
            dataset = self.train_dataset, 
            batch_size = self.batch_size, 
            shuffle = True
            )
    

    def val_dataloader(self):
        return DataLoader(
            dataset = self.val_dataset, 
            batch_size = self.batch_size, 
            shuffle = False
            )
    

    def test_dataloader(self):
        return DataLoader(
            dataset = self.test_dataset, 
            batch_size = self.batch_size, 
            shuffle = False
            )
