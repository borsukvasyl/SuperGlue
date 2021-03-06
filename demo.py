from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from fire import Fire
from kornia.feature import match_smnn
from skimage.io import imread

from superglue.detectors import DoGHardNetDetector, Detections
from superglue.superglue import SuperGlueMatcher


def visualize(
    img0: np.ndarray,
    img1: np.ndarray,
    detections0: Detections,
    detections1: Detections,
    min_x: np.ndarray,
    min_y: np.ndarray,
    rect: np.ndarray,
) -> plt.Figure:
    img1 = cv2.polylines(img1, [np.int32(rect)], True, (255, 0, 0), 3, cv2.LINE_AA)
    vis = np.hstack([img0, img1])

    kpts0 = detections0.kpts
    kpts1 = detections1.kpts
    kpts1 = kpts1.copy()
    kpts1[:, 0] += img0.shape[1]

    fig = plt.figure(figsize=(20, 10))
    plt.imshow(vis)
    for x, y in zip(min_x, min_y):
        plt.plot([kpts0[x][0], kpts1[y][0]], [kpts0[x][1], kpts1[y][1]])
    return fig


def smnn(dets0: Detections, dets1: Detections) -> Tuple[np.ndarray, np.ndarray]:
    inp0 = torch.from_numpy(dets0.kpts)
    inp1 = torch.from_numpy(dets1.kpts)
    vals, idxs = match_smnn(inp0, inp1, 0.9)
    min_x = idxs.detach().numpy()[:, 0]
    min_y = idxs.detach().numpy()[:, 1]
    return min_x, min_y


def ransac(
    img0: np.ndarray,
    detections0: Detections,
    detections1: Detections,
    min_x: np.ndarray,
    min_y: np.ndarray
):
    src_pts = detections0.kpts[min_x]
    dst_pts = detections1.kpts[min_y]
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 10000)
    h, w, _ = img0.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    rectangle = cv2.perspectiveTransform(pts, H)
    return rectangle


def main(img0_path: str, img1_path: str, save_path: str, confidence_thr: float = 0.1, do_smnn: bool = False) -> None:
    detector = DoGHardNetDetector(1024)
    matcher = SuperGlueMatcher(confidence_thr=confidence_thr)
    img0 = imread(img0_path)[..., :3]
    img1 = imread(img1_path)[..., :3]

    detections0 = detector.detect(img0)
    detections1 = detector.detect(img1)
    if do_smnn:
        title = "SMNN"
        min_x, min_y = smnn(detections0, detections1)
    else:
        title = "SuperGlue"
        min_x, min_y, _ = matcher.match(detections0, detections1, img0.shape[:2], img1.shape[:2])

    rect = ransac(img0, detections0, detections1, min_x, min_y)

    fig = visualize(img0, img1, detections0, detections1, min_x, min_y, rect)
    fig.suptitle(title, fontsize=24)
    plt.tight_layout()
    fig.savefig(save_path)


if __name__ == '__main__':
    Fire(main)
