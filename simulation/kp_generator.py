import numpy as np


def generate_keypoints_equal(top_right_point: np.ndarray,
                             x_keypoint_amount: int = 10, y_keypoint_amount: int = 10):
    """
    Generate keypoints equal spread through image
    :param top_right_point:
    :param x_keypoint_amount:
    :param y_keypoint_amount:
    :return:
    """
    x_delta = top_right_point[0] / x_keypoint_amount
    y_delta = top_right_point[1] / y_keypoint_amount
    keypoints = []
    for x_index in range(0, x_keypoint_amount):
        for y_index in range(0, y_keypoint_amount):
            tmp = np.ndarray([3, 1], dtype='int32')

            tmp[0] = int((x_index + 1. / 2) * x_delta)
            tmp[1] = int((y_index + 1. / 2) * y_delta)
            tmp[2] = int(0)

            keypoints.append(tmp)
    return keypoints
