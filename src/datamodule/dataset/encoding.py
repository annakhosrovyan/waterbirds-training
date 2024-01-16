import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class EncodingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, c: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.c = torch.from_numpy(c)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x[index], self.y[index], self.c[index]
