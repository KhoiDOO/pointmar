from torch.utils.data import Dataset
from torch import Tensor

from datasets import load_dataset

import numpy as np
import torch

class ShapeNet(Dataset):
    def __init__(self, root: str, num_points: int = 2048):
        super().__init__()

        if num_points not in [1024]:
            raise ValueError("num_points should be one of 2048, 4096, or 8192 for ShapeNet dataset")

        self._root = root
        self._num_points = num_points
        self._dataset = load_dataset(f"kohido/shapenet_{num_points}pts", cache_dir=self._root)['train']

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tensor:
        points = self._dataset[index]['points']
        points: np.ndarray = np.array(points) # (self._num_points, 3)
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
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