import cv2
import numpy as np


def optical_flow_visualization(img, keypoints, optical_flow) -> np.ndarray:
    res_img = img.copy()
    for (kp, of) in zip(keypoints, optical_flow):
        cv2.drawMarker(
            res_img,
            (int(kp[0]), int(kp[1])),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS
        )
        cv2.line(
            res_img,
            (int(kp[0]), int(kp[1])),
            (int(kp[0] + of[0]), int(kp[1] + of[1])),
            (255, 255, 255),
            thickness=2
        )
        cv2.drawMarker(
            res_img,
            (int(kp[0] + of[0]), int(kp[1] + of[1])),
            (255, 255, 255),
            markerType=cv2.MARKER_DIAMOND
        )
    return res_img