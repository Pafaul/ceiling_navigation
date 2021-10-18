import numpy as np


def convert_keypoints_to_px(keypoints: list, coeffs: list, canvas_size: tuple) -> list:
    """

    :rtype: object
    """
    result_keypoints_px = []
    for kp in keypoints:
        tmp = [0, 0]
        tmp[0] = kp[0] * coeffs[0]
        tmp[1] = kp[1] * coeffs[1]
        tmp[0], tmp[1] = kp[1], canvas_size[0] - kp[0]
        result_keypoints_px.append(tmp)
    return result_keypoints_px


def transform_keypoints(initial_3d_keypoints: list, rotation_matrix: np.ndarray,
                        displacement_vector: np.ndarray) -> list:
    """
    Generate new 3d keypoints from initial 3d keypoints
    3D_2 = rotation_matrix*3D_1 + displacement_vector
    :param initial_3d_keypoints: initial 3d keypoints placement
    :param rotation_matrix: rotation matrix used to rotate keypoints
    :param displacement_vector: displacement vector to apply to keypoints
    :return:
    """
    transformed_keypoints = []
    for kp in initial_3d_keypoints:
        transformed_keypoints.append(np.dot(rotation_matrix, kp) + displacement_vector)

    return transformed_keypoints
