import cv2
import numpy as np


def match_keypoints(kp_1, des_1, kp_2, des_2, matcher):
    """
    Match keypoints
    :param kp_1: keypoints of the first image
    :param des_1: descriptors of the first image keypoints
    :param kp_2: keypoints of the second image
    :param des_2: descriptors of the second image keypoints
    :param matcher: algorithm used for keypoint matching (currently: BF/KNN)
    :return: matched keypoints
    """
    matches = matcher.knnMatch(des_1, des_2, k=2)

    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts1.append(kp_1[m.queryIdx].pt)
            pts2.append(kp_2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2
