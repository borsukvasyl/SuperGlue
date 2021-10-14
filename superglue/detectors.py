from abc import ABC, abstractmethod
from collections import namedtuple

import cv2
import numpy as np


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
        self.sift = cv2.SIFT_create(nfeatures=self.num_features)

    def detect(self, img: np.ndarray) -> Detections:
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        if keypoints is None or descriptors is None:
            return Detections(kpts=np.zeros((0, 2)), desc=np.zeros((0, 128)), meta=np.zeros((0, 1)))

        keypoints, descriptors = keypoints[:self.num_features], descriptors[:self.num_features]
        coords = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
        meta = np.array([[kp.response] for kp in keypoints])
        return Detections(
            kpts=coords,
            desc=descriptors,
            meta=meta,
        )
