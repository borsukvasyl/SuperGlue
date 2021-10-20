from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import cv2
import torch
import kornia.feature as KF
from extract_patches.core import extract_patches


Detections = namedtuple("Detections", ["kpts", "desc", "meta"])


class BaseDetector(ABC):
    def __init__(self, num_features: int):
        self.num_features = num_features

    @abstractmethod
    def detect(self, img: np.ndarray) -> Detections:
        pass


class SIFTDetector(BaseDetector):
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.sift = cv2.SIFT_create(nfeatures=2 * self.num_features)

    def detect(self, img: np.ndarray) -> Detections:
        keypoints, descriptors = self.sift.detectAndCompute(img, None)

        keypoints, descriptors = keypoints[:self.num_features], descriptors[:self.num_features]
        kpts = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
        meta = np.array([[kp.response] for kp in keypoints])
        return Detections(
            kpts=kpts,
            desc=descriptors,
            meta=meta,
        )


class DoGHardNetDetector(BaseDetector):
    def __init__(self, num_features: int):
        super().__init__(num_features)
        self.sift = cv2.SIFT_create(2 * self.num_features)
        self.hardnet = KF.HardNet(True).cpu().eval()

    def detect(self, img: np.ndarray) -> Detections:
        keypoints = self.extract_sift_keypoints(img)
        As = torch.eye(2).view(1, 2, 2).expand(len(keypoints), 2, 2).numpy()
        descriptors = self.extract_descriptors(keypoints, img, As)

        kpts = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
        meta = np.array([[kp.response] for kp in keypoints])
        return Detections(
            kpts=kpts,
            desc=descriptors,
            meta=meta,
        )

    def extract_sift_keypoints(self, img):
        keypoints = self.sift.detect(img, None)
        response = np.array([kp.response for kp in keypoints])
        resp_sort = np.argsort(response)[::-1]
        kpts = [keypoints[i] for i in resp_sort[:self.num_features]]
        return kpts

    def extract_descriptors(self, kpts, img, As):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        patches = np.array(extract_patches((kpts, As), img_gray, 32, 12., 'cv2+A')).astype(np.float32)
        bs = 128
        desc = np.zeros((len(patches), 128))
        for i in range(0, len(patches), bs):
            data_a = torch.from_numpy(patches[i:min(i + bs, len(patches)), :, :]).unsqueeze(1)
            with torch.no_grad():
                out_a = self.hardnet(data_a)
                desc[i:i + bs] = out_a.cpu().detach().numpy()
        return desc
