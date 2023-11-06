import numpy as np
from torch.utils.data import DataLoader
from datamodule.encoding import EncodingDataset


class DataModule:
    def __init__(self, train_path, test_path, val_path, batch_size, *args, **kwargs):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.batch_size = batch_size


    def get_data(self, path):
        data = np.load(path)
        x, y, c = data["embeddings"], data["labels"], data["domains"]
        
        return x.squeeze(), y.squeeze(), c.squeeze()


    def prepare_data(self):
        train_x, train_y, train_c = self.get_data(self.train_path)
        test_x, test_y, test_c = self.get_data(self.test_path)
        val_x, val_y, val_c = self.get_data(self.val_path)

        train_dataset = EncodingDataset(train_x, train_y, train_c)
        val_dataset = EncodingDataset(val_x, val_y, val_c)
        test_dataset = EncodingDataset(test_x, test_y, test_c)

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle = True)
        val_loader = DataLoader(val_dataset, self.batch_size, shuffle = False)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle = False)

        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
