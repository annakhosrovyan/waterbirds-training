import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader


#       ------------------------------------------------------------------------------------------------------------
#       -------------------------------Waterbirds Image Dataset Loading and Processing------------------------------
#       ------------------------------------------------------------------------------------------------------------


class WaterbirdsDatasetLoader:
    def __init__(self, root_dir, dataset_name, download = True):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.download = download

    def get_dataset(self):
        dataset = get_dataset(root_dir = self.root_dir, dataset = self.dataset_name, download = self.download)
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

    def get_test_data(self):
        dataset = self.get_dataset()
        test_dataset = dataset.get_subset(
            "test",
            transform=transforms.Compose(
                [transforms.Resize((448, 448)), transforms.ToTensor()]
            ),
        )
        return test_dataset

    def load_data(self, batch_size):
        dataset = self.get_dataset()
        train_dataset = self.get_train_data()
        test_dataset = self.get_test_data()

        train_loader = get_train_loader("standard", train_dataset, batch_size=batch_size)
        test_loader = get_eval_loader("standard", test_dataset, batch_size=batch_size)

        return train_loader, test_loader


#       ------------------------------------------------------------------------------------------------------------
#       -----------------------------Representation Data Loading and Processing----------------------------
#       ------------------------------------------------------------------------------------------------------------


class RepresentationDatasetLoader:
    def __init__(self, train_path, test_path, val_path, batch_size):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.batch_size = batch_size

    def get_data(self, path):
        data = np.load(path)
        x, y, c = data["embeddings"], data["labels"], data["domains"]
        return x.squeeze(), y.squeeze(), c.squeeze()

    def load_data(self):
        train_x, train_y, train_c = self.get_data(self.train_path)
        test_x, test_y, test_c = self.get_data(self.test_path)
        val_x, val_y, val_c = self.get_data(self.val_path)

        train_dataset = EncodingDataset(train_x, train_y, train_c)
        test_dataset = EncodingDataset(test_x, test_y, test_c)
        val_dataset = EncodingDataset(val_x, val_y, val_c)

        train_loader = DataLoader(train_dataset, self.batch_size, shuffle = True)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle = False)

        return train_loader, test_loader, train_dataset, test_dataset, val_dataset


class EncodingDataset(Dataset):
    def __init__(self, x, y, c):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.c = torch.from_numpy(c)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.c[index]