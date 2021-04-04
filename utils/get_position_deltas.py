import cv2
import cv2
import numpy as np

def get_position_deltas(keypoints_set_1: np.array, keypoints_set_2: np.array, camera_matrix: np.array):
    retval, mask = cv2.findEssentialMat(keypoints_set_1, keypoints_set_2, camera_matrix)
    retval, R, t, mask = cv2.recoverPose(retval, keypoints_set_1, keypoints_set_2, camera_matrix, mask=mask)
    return R, t