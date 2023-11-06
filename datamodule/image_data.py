import torch
import torchvision.transforms as transforms

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


class DataModule:
    def __init__(self, root_dir, batch_size, download = True, *args, **kwargs):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.download = download


    def get_dataset(self): 
        dataset = get_dataset(root_dir = self.root_dir, dataset = "waterbirds", download = self.download)
      
        return dataset


    def get_train_data(self):
        dataset = self.get_dataset() 
        train_dataset = dataset.get_subset(
            "train",
            transform=transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            ),
        )

        return train_dataset
    

    def get_val_data(self):
        dataset = self.get_dataset()
        val_dataset = dataset.get_subset(
            "val",
            transform=transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            ),
        )

        return val_dataset


    def get_test_data(self):
        dataset = self.get_dataset()
        test_dataset = dataset.get_subset(
            "test",
            transform=transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            ),
        )

        return test_dataset


    def prepare_data(self):
        dataset = self.get_dataset()
        train_dataset = self.get_train_data()
        val_dataset = self.get_val_data()
        test_dataset = self.get_test_data()

        train_loader = get_train_loader("standard", train_dataset, batch_size = self.batch_size)
        val_loader = get_eval_loader("standard", val_dataset, batch_size = self.batch_size)
        test_loader = get_eval_loader("standard", test_dataset, batch_size = self.batch_size)

        return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset