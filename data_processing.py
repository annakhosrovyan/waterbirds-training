import torch
import numpy as np
import torchvision.transforms as transforms 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from wilds import get_dataset
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.data_loaders import get_eval_loader


#       ####################################

#       Waterbirds Images Dataset Processing

#       ####################################


def get_waterbirds_dataset():
    dataset = get_dataset(root_dir = "C:/Users/User/Desktop/MyProjects/waterbirds_training/data", 
                          dataset = "waterbirds", download = True)

    return dataset


def get_train_data():
    dataset = get_waterbirds_dataset()
    train_dataset = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    return train_dataset


def get_test_data():
    dataset = get_waterbirds_dataset()
    test_dataset = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )
    return test_dataset


def load_waterbirds_images_data(batch_size):    
    dataset = get_waterbirds_dataset()
 
    train_dataset = get_train_data()
    test_dataset = get_test_data()

    train_loader = get_train_loader("standard", train_dataset, batch_size = batch_size)
    test_loader = get_eval_loader("standard", test_dataset, batch_size = batch_size)

    return train_loader, test_loader



#       ###########################################

#       ResNet50 Representation Dataset Processing

#       ###########################################


class EncodingDataset(Dataset):
    def __init__(self, x, y, c):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.c = torch.from_numpy(c)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
       return self.x[index], self.y[index], self.c[index]
    

def get_data(train_path):
    data = np.load(train_path)
    x, y, c = data["embeddings"], data["labels"], data["domains"]

    return x.squeeze(), y.squeeze(), c.squeeze()
    

def load_resnet50_representation_data(train_path, test_path, batch_size):
    train_x, train_y, train_c = get_data(train_path)
    test_x, test_y, test_c  = get_data(test_path)

    train_dataset = EncodingDataset(train_x, train_y, train_c)
    test_dataset = EncodingDataset(test_x, test_y, test_c)

    train_loader = DataLoader(train_dataset, batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle = False)

    return train_loader, test_loader,  train_dataset, test_dataset






