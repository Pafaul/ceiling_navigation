import numpy as np


def get_keypoints_from_image(img: np.array, detector):
    """
    Extract keypoints from the image
    :param img: image to extract keypoints from
    :param detector: detector used for keypoints extraction
    :return:
    """
    return detector.detectAndCompute(img, None)
