import os
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from superglue.detectors import Detections


def preprocess_keypoints(
        kpts: np.ndarray, desc: np.ndarray, meta: np.ndarray, img_size: Tuple[int, int], normalize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = img_size
    kpts = (kpts - (width / 2, height / 2)) / max(width, height)
    kpts = np.concatenate([kpts, meta], axis=1)
    kpts = np.transpose(kpts)
    if normalize:
        desc = desc / 255.
    desc = np.transpose(desc)
    return kpts.astype(np.float32), desc.astype(np.float32)


class SuperGlueMatcher:
    def __init__(self, jit_path: str = "SuperGlue-DoGHardNet-200000steps.pt", confidence_thr: float = 0.1):
        self.model = self._load_model(jit_path)
        self.confidence_thr = confidence_thr

    def match(self, dets0: Detections, dets1: Detections, img0_size: Tuple[int, int], img1_size: Tuple[int, int]):
        kpts0, desc0 = preprocess_keypoints(dets0.kpts, dets0.desc, dets0.meta, img0_size)
        kpts1, desc1 = preprocess_keypoints(dets1.kpts, dets1.desc, dets1.meta, img1_size)

        kpts0 = torch.from_numpy(kpts0)[None]
        desc0 = torch.from_numpy(desc0)[None]
        kpts1 = torch.from_numpy(kpts1)[None]
        desc1 = torch.from_numpy(desc1)[None]

        with torch.no_grad():
            scores = self.model(kpts0, desc0, kpts1, desc1)
        scores = scores.exp().numpy()[0]

        x, y = linear_sum_assignment(scores, maximize=True)
        mask = (x < kpts0.shape[2]) & (y < kpts1.shape[2]) & (scores[x, y] > self.confidence_thr)
        x, y = x[mask], y[mask]
        return x, y, scores

    @staticmethod
    def _load_model(jit_path: str) -> torch.jit.ScriptModule:
        lib_path = os.path.dirname(__file__)
        path = os.path.join(lib_path, "checkpoints", jit_path)
        return torch.jit.load(path, map_location="cpu")
