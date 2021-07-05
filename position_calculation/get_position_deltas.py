from math import atan2
from lsm.least_square_method import seq_lsm

import cv2
import numpy as np

# TODO: rewrite this module
def get_position_deltas(pts1: np.array, pts2: np.array, camera_matrix: np.array, config={}):
    try:
        E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
        retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)
        X, mask, error_EV, error_DP = seq_lsm(pts1, pts2, R)
        if len(X) == 0:
            return [], []

        while len(pts1) != mask.count(True):
            pts1 = np.int32(pts1[mask])
            pts2 = np.int32(pts2[mask])

            E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
            if E is None or len(E) != 3:
                return [], []
            retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)

            X, mask, error_EV, error_DP = seq_lsm(pts1, pts2, R)
            if len(X) == 0:
                return [], []

        angles = [
            atan2(R[2][1], R[2][2]),
            atan2(-R[2][0], (R[2][1] ** 2 + R[2][2] ** 2) ** 0.5),
            atan2(R[1][0], R[0][0])
        ]

        if 'angle_threshold' in config:
            angles = [angle if abs(angle) > config['angle_threshold'] else 0 for angle in angles]

        if 'dist_threshold' in config:
            dists = [dist if abs(dist) > config['dist_threshold'] else 0 for dist in X[0:3]]

        X = np.append(X, angles)
        X = np.append(X, [0])  # for delta t
        error_DP = np.append(error_DP, [0, 0, 0, 0])  # for angles and delta t
        return X, error_DP
    except Exception as e:
        print(e)
        return [], []
