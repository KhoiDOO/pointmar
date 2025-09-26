from torch.utils.data import Dataset
from torch import Tensor

from datasets import load_dataset

from scipy.spatial.transform import Rotation

import numpy as np
import torch
import random

class ShapeNet(Dataset):
    def __init__(
            self, 
            root: str, 
            num_points: int = 1024,
            augment: bool = False,
            scale_min: float = 0.75,
            scale_max: float = 0.95,
            rotation: float = 180.0,
        ) -> None:
        super().__init__()

        if num_points not in [1024]:
            raise ValueError("num_points should be 1024 for ShapeNet dataset")

        self._root = root
        self._num_points = num_points
        self._dataset = load_dataset(f"kohido/shapenet_{num_points}pts", cache_dir=self._root)['train']

        self._augment = augment
        self._scale_min = scale_min
        self._scale_max = scale_max
        self._rotation = rotation

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tensor:
        points = self._dataset[index]['points']
        points: np.ndarray = np.array(points) # (self._num_points, 3)
        points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
        if self.augment:
            points = self.augment_func(points, self.scale_min, self.scale_max, self.rotation)
        # scale to a [-0.5, 0.5] cube
        points = points - np.mean(points, axis=0, keepdims=True)
        max_abs = np.max(np.abs(points))
        points = points / (2 * max_abs)

        points: Tensor = torch.from_numpy(points) 
        points = points.float()
        # points = points.permute(1, 0)
        return points

    @staticmethod
    def augment_func(points: np.ndarray, scale_min: float, scale_max: float, rotation: float) -> Tensor:

        for i in range(3):
            # Generate a random scale factor
            scale = random.uniform(scale_min, scale_max)

            # independently applied scaling across each axis of vertices
            points[:, i] *= scale
        
        if rotation != 0.:        
            rotate_upright = random.random() < 0.3
            
            if rotate_upright:
                rotation_options = [0.5 * np.pi, -0.5 * np.pi]
                
                # Randomly choose rotation angles for x and y axes
                rot_x = random.choice(rotation_options)
                rot_y = random.choice(rotation_options)
                case = random.choice([1, 2])
                
                # Apply the rotation based on the chosen case
                if case == 1:
                    rotation_obj = Rotation.from_rotvec([rot_x, 0, 0])
                    points = rotation_obj.apply(points)
                elif case == 2:
                    rotation_obj = Rotation.from_rotvec([0, rot_y, 0])
                    points = rotation_obj.apply(points)

            rot_z = random.uniform(-1, 1) * np.pi * 180
            angles = np.array([0, 0, rot_z])
            rotation_obj = Rotation.from_rotvec(angles)
            points = rotation_obj.apply(points)
        return points

    @property
    def root(self) -> str:
        return self._root

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def augment(self) -> bool:
        return self._augment

    @property
    def scale_min(self) -> float:
        return self._scale_min

    @property
    def scale_max(self) -> float:
        return self._scale_max

    @property
    def rotation(self) -> float:
        return self._rotation