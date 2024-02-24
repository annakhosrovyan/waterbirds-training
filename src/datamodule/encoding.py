import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.datamodule.dataset import EncodingDataset

from enum import Enum
from typing import Tuple, List

class TrainingStage(Enum):
    ERM = 1
    REWEIGTING = 2


class EncodingDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_path: str,
                 test_path: str,
                 val_path: str,
                 batch_size: int,
                 training_type: str,
                 *args, **kwargs
                 ):
        super().__init__()
        self._stage = TrainingStage.ERM

        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.batch_size = batch_size

        self.training_type = training_type

    def get_data(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.load(path)
        x, y, c = data["embeddings"], data["labels"], data["domains"]

        return x.squeeze(), y.squeeze(), c.squeeze()

    def setup(self, stage=None) -> Tuple[EncodingDataset, EncodingDataset, EncodingDataset, EncodingDataset]:
        train_x, train_y, train_c = self.get_data(self.train_path)
        test_x, test_y, test_c = self.get_data(self.test_path)
        val_x, val_y, val_c = self.get_data(self.val_path)

        self.train_dataset = EncodingDataset(train_x, train_y, train_c)
        self.val_dataset = EncodingDataset(val_x, val_y, val_c)
        self.test_dataset = EncodingDataset(test_x, test_y, test_c)

        self.first_stage_dataset, self.second_stage_dataset = self.split_train_dataset(self.train_dataset)

        return self.first_stage_dataset, self.second_stage_dataset, self.val_dataset, self.test_dataset
    
    
    def split_train_dataset(self, train_dataset, first_stage_ratio=0.8) -> Tuple[EncodingDataset, EncodingDataset]:
        first_stage_data_size = int(len(train_dataset) * first_stage_ratio)
        second_stage_data_size = len(train_dataset) - first_stage_data_size
       
        return torch.utils.data.random_split(self.train_dataset, 
                                            [first_stage_data_size, 
                                             second_stage_data_size])


    def train_dataloader(self) -> DataLoader:
        data = self.first_stage_dataset if self._stage == TrainingStage.ERM else self.second_stage_dataset
        if self.training_type == 'standard':
            data = torch.utils.data.ConcatDataset([self.first_stage_dataset, self.second_stage_dataset])
            
        return DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )


    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def change_to_2nd_stage(self, model, gamma):
        self._stage = TrainingStage.REWEIGTING

        model.eval()
        model.cuda()
        logits = []
        ys = []
        for x, y, c in self.train_dataloader():
            y_hat = model(x.to("cuda")).detach().cpu()
            logits.append(y_hat)
            ys.append(y)
        logits = torch.cat(logits)
        ys = torch.cat(ys)

        weights = self.compute_afr_weights(logits, ys, gamma)

        return weights

    @staticmethod
    def compute_afr_weights(erm_logits, class_label, gamma) -> torch.Tensor:
        with torch.no_grad():
            p = erm_logits.softmax(-1)
        y_onehot = torch.zeros_like(erm_logits).scatter_(-1, class_label.unsqueeze(-1), 1)
        p_true = (p * y_onehot).sum(-1)
        weights = (-gamma * p_true).exp()
        n_classes = torch.unique(class_label).numel()
        class_count = []
        for y in range(n_classes):
            class_count.append((class_label == y).sum())
        for y in range(0, n_classes):
            weights[class_label == y] *= 1 / class_count[y]
        weights /= weights.sum()
        return weights
