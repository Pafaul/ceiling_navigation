import numpy as np
import math
import cv2

from simulation.camera import Camera


def get_keypoints_both_pictures(all_kp: list, kp1: list, kp2: list, mask1: list, mask2: list) -> (list, list, list):
    visible_kp_1 = []
    visible_kp_2 = []
    real_visible_kp = []
    mask = []
    for (is_visible_1, is_visible_2, kp_1, kp_2, real_kp) in zip(mask1, mask2, kp1, kp2, all_kp):
        if is_visible_1 and is_visible_2:
            mask.append(True)
            visible_kp_1.append(kp_1)
            visible_kp_2.append(kp_2)
            real_visible_kp.append(real_kp)
        else:
            mask.append(False)

    visible_kp_1 = np.int32(visible_kp_1)
    visible_kp_2 = np.int32(visible_kp_2)
    return visible_kp_1, visible_kp_2, real_visible_kp, mask


def get_abs_diff_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    r = abs(R1 - R2)
    return sum(sum(r))


def calculate_angles(R: np.ndarray) -> list:
    def rad_to_deg(x):
        return x * 180 / math.pi

    return [
        rad_to_deg(math.atan(R[2, 1] / R[2, 2])),
        rad_to_deg(-math.asin(R[2, 0])),
        rad_to_deg(math.atan(R[1, 0] / R[0, 0]))
    ]


def calculate_obj_rotation_matrix(
        previous_kp: np.int32,
        current_kp: np.int32,
        camera: Camera,
        rotation_matrix: np.ndarray
):
    E, mask = cv2.findEssentialMat(previous_kp, current_kp, camera.internal_matrix, method=cv2.LMEDS, threshold=0.1)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    tmp_rotation_matrix_1 = np.dot(R1, rotation_matrix)
    tmp_rotation_matrix_2 = np.dot(R2, rotation_matrix)
    # print(f'angles1: {calculate_angles(tmp_rotation_matrix_1)}, angles2: {calculate_angles(tmp_rotation_matrix_2)}')
    delta1 = get_abs_diff_mat(tmp_rotation_matrix_1, rotation_matrix)
    delta2 = get_abs_diff_mat(tmp_rotation_matrix_2, rotation_matrix)

    # print(f'delta1: {1}, delta2: {2}', delta1, delta2)

    if delta1 < delta2:
        rotation_matrix = tmp_rotation_matrix_1.copy()
    else:
        rotation_matrix = tmp_rotation_matrix_2.copy()

    return rotation_matrix


def calculate_angles_delta(real_angles, angles) -> list:
    deltas = [(
        real_angle[0] + angle[0],
        real_angle[1] + angle[1],
        real_angle[2] - angle[2]
    ) for (real_angle, angle) in zip(real_angles, angles)]
    return deltas
