import glob

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset


def preprocess(kpts, scores, descriptors, img_size):
    height, width = img_size
    kpts = (kpts - (width / 2, height / 2)) / max(width, height)
    kpts = np.concatenate([kpts, scores[..., None]], axis=1)
    kpts = np.transpose(kpts)
    descriptors = descriptors / 255.
    descriptors = np.transpose(descriptors)
    return kpts.astype(np.float32), descriptors.astype(np.float32)


class SuperGlueDataset(Dataset):
    def __init__(
            self,
            image_glob: str,
            warping_ratio: float = 0.2,
            num_features: int = 1024,
            return_metadata: bool = False,
    ):
        self.image_paths = sorted(glob.glob(image_glob))
        self.warping_ratio = warping_ratio
        self.num_features = num_features
        self.return_metadata = return_metadata
        self.sift = None

    def __getitem__(self, index):
        if self.sift is None:
            self.sift = cv2.SIFT_create(nfeatures=self.num_features)

        orig_image = self.read_image(self.image_paths[index])
        transform_matrix = self.get_transformation_matrix(shape=orig_image.shape, warping_ratio=self.warping_ratio)
        warped_image = self.warp_image(orig_image, transform_matrix)
        orig_features = self.get_features(orig_image, num_features=self.num_features)
        warped_features = self.get_features(warped_image, num_features=self.num_features)
        projected_kpts = self.warp_keypoints(orig_features["kpts"], transform_matrix)
        matches = self.get_matches(projected_kpts, warped_features["kpts"])

        img_size = orig_image.shape[:2]
        kpts0, desc0 = preprocess(
            orig_features["kpts"], orig_features["scores"], orig_features["descs"], img_size
        )
        kpts1, desc1 = preprocess(
            warped_features["kpts"], warped_features["scores"], warped_features["descs"], img_size
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
                    orig_kpts=orig_features["kpts"],
                    orig_descs=orig_features["descs"],
                    orig_scores=orig_features["scores"],
                    warped_kpts=warped_features["kpts"],
                    warped_descs=warped_features["descs"],
                    warped_scores=warped_features["scores"],
                    warped_image=warped_image,
                    projected_kpts=projected_kpts,
                    transform_matrix=transform_matrix,
                )
            )
        return outputs

    def __len__(self):
        return len(self.image_paths)

    def get_features(self, image: np.array, num_features: int):
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        keypoints, descriptors = keypoints[:num_features], descriptors[:num_features]
        keypoints_value = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
        keypoints_score = np.array([kp.response for kp in keypoints])
        return dict(
            kpts=keypoints_value,
            scores=keypoints_score,
            descs=descriptors
        )

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
        return cv2.perspectiveTransform(keypoints.reshape((1, -1, 2)), transform_matrix)[0]

    @staticmethod
    def read_image(path: str) -> np.array:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
