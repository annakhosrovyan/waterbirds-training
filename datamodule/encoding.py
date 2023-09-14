import torch
from torch.utils.data import Dataset

class EncodingDataset(Dataset):
    def __init__(self, x, y, c):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.c = torch.from_numpy(c)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.c[index]