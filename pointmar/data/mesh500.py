from torch.utils.data import Dataset
from torch import Tensor

from datasets import load_dataset

import numpy as np
import torch

class Mesh500(Dataset):
    def __init__(self, root: str, num_points: int = 1024):
        super().__init__()

        if num_points not in [1024, 2048, 4096]:
            raise ValueError("num_points should be one of 1024, 2048, or 4096 for Mesh500 dataset")

        self._root = root
        self._num_points = num_points
        if self._num_points == 2048:
            self._dataset = load_dataset(f"kohido/mesh500_4096pts", cache_dir=self._root)['train']['points']
        elif self._num_points == 4096:
            self._dataset = load_dataset(f"kohido/mesh500_4096pts", cache_dir=self._root)['train']['points']
        else: # 1024
            self._dataset = load_dataset(f"kohido/mesh500_1024pts", cache_dir=self._root)['train']['points']

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tensor:
        points = self._dataset[index]
        points: np.ndarray = np.array(points) # (self._num_points, 3)
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
        if self._num_points == 2048:
            # only take even indices
            points = points[::2]
        # scale to a [-0.5, 0.5] cube
        points = points - np.mean(points, axis=0, keepdims=True)
        max_abs = np.max(np.abs(points))
        points = points / (2 * max_abs)

        points: Tensor = torch.from_numpy(points) 
        points = points.float()
        # points = points.permute(1, 0)
        return points

    @property
    def root(self) -> str:
        return self._root