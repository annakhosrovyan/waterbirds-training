import numpy as np
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset

class CustomizedWaterbirdsDataset(WaterbirdsDataset):
    weights = None

    def __init__(self, rw_ratio = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rw_ratio = rw_ratio

        self._make_reweigting_set()
        
        self.cache = {}

    def _make_reweigting_set(self):

        train_indices = np.where(self._split_array == self.split_dict['train'])[0]
        num_to_change = int(len(train_indices) * self.rw_ratio)

        selected_indices = np.random.choice(train_indices, num_to_change, replace=False)
        self._split_array[selected_indices] = 3
        self._split_names = {'train': 'Train For ERM',
                             'val': 'Validation',
                             'test': 'Test',
                             'train_rw': 'Train for reweighting'}
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'train_rw': 3}

    def __getitem__(self, idx):

        x, y, metadata = super().__getitem__(idx)
        x, y, c = x, y, metadata[0]
        if self.weights is None or idx not in self.weights:
            return x, y, c
        else:
            w = self.weights[idx]
            # print(w, c)
            return x, y, w
