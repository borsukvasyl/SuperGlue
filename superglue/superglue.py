from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def preprocess_keypoints(kpts: np.ndarray, desc: np.ndarray, meta: np.ndarray, img_size: Tuple[int, int]):
    height, width = img_size
    kpts = (kpts - (width / 2, height / 2)) / max(width, height)
    kpts = np.concatenate([kpts, meta], axis=1)
    kpts = np.transpose(kpts)
    descriptors = desc / 255.
    descriptors = np.transpose(descriptors)
    return kpts.astype(np.float32), descriptors.astype(np.float32)


class SuperGlueMatcher:
    def __init__(self, jit_path: str):
        self.model = torch.jit.load(jit_path, map_location="cpu")

    def match(self, kpts0: np.ndarray, desc0: np.ndarray, kpts1: np.ndarray, desc1: np.ndarray):
        kpts0 = torch.from_numpy(kpts0)[None]
        desc0 = torch.from_numpy(desc0)[None]
        kpts1 = torch.from_numpy(kpts1)[None]
        desc1 = torch.from_numpy(desc1)[None]
        scores = self.model(kpts0, desc0, kpts1, desc1)
        scores_numpy = scores.numpy()[0]
        x, y = linear_sum_assignment(scores_numpy, maximize=True)
        return x, y
