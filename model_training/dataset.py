import glob

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from numpy.random import randint
from torch.utils.data import Dataset

from model_training.utils import create_logger
from superglue.detectors import DoGHardNetDetector
from superglue.superglue import preprocess_keypoints


logger = create_logger("Dataset")


class SuperGlueDataset(Dataset):
    def __init__(
        self,
        image_glob: str,
        warping_ratio: float = 0.25,
        num_features: int = 1024,
        return_metadata: bool = False,
    ):
        self.image_paths = sorted(glob.glob(image_glob))
        self.warping_ratio = warping_ratio
        self.num_features = num_features
        self.return_metadata = return_metadata
        self.detector = None

    def __getitem__(self, index):
        # distributed training fix
        if self.detector is None:
            self.detector = DoGHardNetDetector(self.num_features)

        try:
            return self.get(index)
        except Exception as e:
            logger.error(f"Failed to get {index}: {e}")
            return self.get(0)

    def get(self, index):
        # read image and transform it
        orig_image = self.read_image(self.image_paths[index])
        transform_matrix = self.get_transformation_matrix(shape=orig_image.shape, warping_ratio=self.warping_ratio)
        warped_image = self.warp_image(orig_image, transform_matrix)

        # detect and match keypoints
        orig_features = self.detector.detect(orig_image)
        warped_features = self.detector.detect(warped_image)
        projected_kpts = self.warp_keypoints(orig_features.kpts, transform_matrix)
        matches = self.get_matches(projected_kpts, warped_features.kpts)

        img_size = orig_image.shape[:2]
        kpts0, desc0 = preprocess_keypoints(
            orig_features.kpts, orig_features.desc, orig_features.meta, img_size
        )
        kpts1, desc1 = preprocess_keypoints(
            warped_features.kpts, warped_features.desc, warped_features.meta, img_size
        )

        outputs = dict(
            kpts0=kpts0,
            desc0=desc0,
            kpts1=kpts1,
            desc1=desc1,
            matches=matches,
        )
        if self.return_metadata:
            outputs.update(
                dict(
                    orig_image=orig_image,
                    orig_kpts=orig_features.kpts,
                    orig_descs=orig_features.desc,
                    orig_scores=orig_features.meta,
                    warped_kpts=warped_features.kpts,
                    warped_descs=warped_features.desc,
                    warped_scores=warped_features.meta,
                    warped_image=warped_image,
                    projected_kpts=projected_kpts,
                    transform_matrix=transform_matrix,
                )
            )
        return outputs

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def get_transformation_matrix(shape: np.array, warping_ratio: float) -> np.array:
        height, width = shape[:2]
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        warping_value = warping_ratio * (height * width) ** 0.5
        warp = np.array([
            [randint(0, warping_value), randint(0, warping_value)],
            [randint(-warping_value, 0), randint(0, warping_value)],
            [randint(-warping_value, 0), randint(-warping_value, 0)],
            [randint(0, warping_value), randint(-warping_value, 0)],
        ], dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(corners + warp, corners)
        return transform_matrix

    @staticmethod
    def warp_image(image: np.array, transform_matrix: np.array) -> np.array:
        warped_image = cv2.warpPerspective(src=image, M=transform_matrix, dsize=(image.shape[1], image.shape[0]))
        return warped_image

    @staticmethod
    def warp_keypoints(keypoints: np.array, transform_matrix: np.array) -> np.array:
        return cv2.perspectiveTransform(keypoints.reshape((1, -1, 2)), transform_matrix)[0]

    @staticmethod
    def get_matches(keypoints_a: np.array, keypoints_b: np.array, threshold: float = 3.) -> np.array:
        dists = cdist(keypoints_a, keypoints_b)
        min_x, min_y = linear_sum_assignment(dists)
        mask = dists[min_x, min_y] < threshold
        min_x, min_y = min_x[mask], min_y[mask]

        num_keypoints_a = len(keypoints_a)
        num_keypoints_b = len(keypoints_b)
        unmatched_x = np.setdiff1d(np.arange(num_keypoints_a), min_x)
        unmatched_y = np.setdiff1d(np.arange(num_keypoints_b), min_y)
        matches = np.zeros((num_keypoints_a + 1, num_keypoints_b + 1))
        matches[min_x, min_y] = 1
        matches[-1, unmatched_y] = 1
        matches[unmatched_x, -1] = 1
        return matches

    @staticmethod
    def read_image(path: str) -> np.array:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
