import cv2
import numpy as np

def get_keypoints_from_image(img: np.array, detector):
    return detector.detectAndCompute(img, None)