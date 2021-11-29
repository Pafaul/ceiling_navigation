from typing import Tuple

import numpy as np
from cv2 import cv2

from camera_navigation.detectors import BasicDetector, SIFTDetector


class BasicMatcher:
    def match_keypoints(self, kp_1, des_1, kp_2, des_2) -> Tuple[np.int32, np.int32]:
        raise NotImplementedError()

    def get_raw_matches(self, kp_1, des_1, kp_2, des_2):
        raise NotImplementedError()

    @staticmethod
    def _distance_check(matches, kp_1, kp_2) -> Tuple[np.int32, np.int32]:
        pts_1 = []
        pts_2 = []

        # for m, n in matches:
            # if m.distance < 0.75 * n.distance:
        for m in matches:
            pts_1.append(kp_1[m.queryIdx].pt)
            pts_2.append(kp_2[m.trainIdx].pt)

        pts_1_ret = np.int32(pts_1)
        pts_2_ret = np.int32(pts_2)

        return pts_1_ret, pts_2_ret


class BFFMatcher(BasicMatcher):
    def __init__(self, detector: BasicDetector):
        norm = cv2.NORM_L2 if isinstance(detector, SIFTDetector) else cv2.NORM_HAMMING
        self.matcher = cv2.BFMatcher_create(normType=norm, crossCheck=True)

    def get_raw_matches(self, kp_1, des_1, kp_2, des_2):
        return self.matcher.match(des_1, des_2)

    def match_keypoints(self, kp_1, des_1, kp_2, des_2) -> Tuple[np.int32, np.int32]:
        matches = self.matcher.match(des_1, des_2)
        return self._distance_check(matches, kp_1, kp_2)
