import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class SuperGlueDataset(torch.utils.data.Dataset):
    def __init__(self, warping_ratio: float = 0.2, num_features: int = 1024):
        self.warping_ratio = warping_ratio
        self.num_features = num_features
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.num_features)
        self.images = ["building.jpg"]

    def __getitem__(self, index):
        orig_image = self.read_image(self.images[index])
        transform_matrix = self.get_transformation_matrix(shape=orig_image.shape, warping_ratio=self.warping_ratio)
        warped_image = self.warp_image(orig_image, transform_matrix)
        orig_features = self.get_features(orig_image, num_features=self.num_features)
        warped_features = self.get_features(warped_image, num_features=self.num_features)
        projected_kpts = self.warp_keypoints(orig_features["kpts"], transform_matrix)
        matches = self.get_matches(projected_kpts, warped_features["kpts"], num_features=self.num_features)
        return dict(
            orig_image=orig_image,
            orig_kpts=orig_features["kpts"],
            orig_descs=orig_features["descs"],
            orig_scores=orig_features["scores"],
            warped_kpts=warped_features["kpts"],
            warped_descs=warped_features["descs"],
            warped_scores=warped_features["scores"],
            warped_image=warped_image,
            projected_kpts=projected_kpts,
            M=transform_matrix,
            matches=matches
        )

    def __len__(self):
        return len(self.images)

    def get_features(self, image: np.array, num_features: int = 64):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        keypoints, descriptors = keypoints[:num_features], descriptors[:num_features]

        keypoints_value = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
        keypoints_score = np.array([kp.response for kp in keypoints])

        keypoints_value = keypoints_value.reshape((1, -1, 2))
        descriptors = np.transpose(descriptors / 256.)
        return dict(
            kpts=keypoints_value,
            scores=keypoints_score,
            descs=descriptors
        )

    @staticmethod
    def get_matches(
            keypoints_a: np.array, keypoints_b: np.array, threshold: float = 3., num_features: int = 64
    ) -> np.array:
        dists = cdist(keypoints_a[0], keypoints_b[0])
        min_x, min_y = linear_sum_assignment(dists)
        mask = dists[min_x, min_y] < threshold
        min_x, min_y = min_x[mask], min_y[mask]
        matches = np.full((num_features,), -1)
        for x, y in zip(min_x, min_y):
            matches[x] = y
        return matches

    @staticmethod
    def get_transformation_matrix(shape: np.array, warping_ratio: float) -> np.array:
        height, width = shape[:2]
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warping_value = warping_ratio * (height * width) ** 0.5
        warp = np.random.randint(-warping_value, warping_value, size=(4, 2)).astype(np.float32)
        transform_matrix = cv2.getPerspectiveTransform(corners, corners + warp)
        return transform_matrix

    @staticmethod
    def warp_image(image: np.array, transform_matrix: np.array) -> np.array:
        warped_image = cv2.warpPerspective(src=image, M=transform_matrix, dsize=(image.shape[1], image.shape[0]))
        return warped_image

    @staticmethod
    def warp_keypoints(keypoints: np.array, transform_matrix: np.array) -> np.array:
        return cv2.perspectiveTransform(keypoints, transform_matrix)

    @staticmethod
    def read_image(path: str) -> np.array:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
