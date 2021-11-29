import numpy as np
from cv2 import cv2
import math


def calculate_angles(R: np.ndarray) -> list:
    def rad_to_deg(x):
        return x * 180 / math.pi

    return [
        rad_to_deg(math.atan(R[2, 1] / R[2, 2])),
        rad_to_deg(-math.asin(R[2, 0])),
        rad_to_deg(math.atan(R[1, 0] / R[0, 0]))
    ]


def get_abs_diff_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    r = abs(R1 - R2)
    return sum(sum(r))


def calculate_obj_rotation_matrix(
        previous_kp: np.int32,
        current_kp: np.int32,
        camera_internal_matrix: np.ndarray,
        rotation_matrix: np.ndarray
):
    E, mask = cv2.findEssentialMat(previous_kp, current_kp, camera_internal_matrix, method=cv2.LMEDS, threshold=0.1)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    tmp_rotation_matrix_1 = np.dot(R1, rotation_matrix)
    tmp_rotation_matrix_2 = np.dot(R2, rotation_matrix)

    delta1 = get_abs_diff_mat(tmp_rotation_matrix_1, rotation_matrix)
    delta2 = get_abs_diff_mat(tmp_rotation_matrix_2, rotation_matrix)

    if delta1 < delta2:
        rotation_matrix = tmp_rotation_matrix_1.copy()
    else:
        rotation_matrix = tmp_rotation_matrix_2.copy()

    return rotation_matrix


def px_to_obj(
        keypoints: list,
        camera_height: float,
        f: float,
        resolution: list,
        px_mm: list,
        rotation_matrix: np.ndarray
) -> list:
    r = rotation_matrix.copy()
    u = resolution[0] / 2 * px_mm[0]
    v = resolution[1] / 2 * px_mm[1]
    kp_obj = []
    for kp in keypoints:
        obj_position = np.ndarray([3, 1])
        obj_position[2] = 0

        obj_position[0] = 0 + (obj_position[2] - camera_height) * \
            (r[0, 0] * (kp[0] * px_mm[0] - u) + r[0, 1] * (kp[1] * px_mm[1] - v) - r[0, 2] * f) / \
            (r[2, 0] * (kp[0] * px_mm[0] - u) + r[2, 1] * (kp[1] * px_mm[1] - v) - r[2, 2] * f)

        obj_position[1] = 0 + (obj_position[2] - camera_height) * \
            (r[1, 0] * (kp[0] * px_mm[0] - u) + r[1, 1] * (kp[1] * px_mm[1] - v) - r[1, 2] * f) / \
            (r[2, 0] * (kp[0] * px_mm[0] - u) + r[2, 1] * (kp[1] * px_mm[1] - v) - r[2, 2] * f)
        kp_obj.append(obj_position)

    return kp_obj


def lsm(A, B):
    return np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose()), B)


def prepare_data_for_lsm(keypoints_obj: list, px_keypoints: list, r: np.ndarray,
                         dz: float, f: float, px_mm: list, resolution: list):
    coefficients = []
    u = resolution[0] / 2 * px_mm[0]
    v = resolution[1] / 2 * px_mm[1]
    for (kp_obj, kp_px) in zip(keypoints_obj, px_keypoints):
        cx1 = (u - kp_px[0] * px_mm[0]) / f
        cy1 = (v - kp_px[1] * px_mm[1]) / f
        cx2 = \
            kp_obj[0] * (r[0, 0] - r[0, 2] * cx1) + \
            kp_obj[1] * (r[1, 0] - r[1, 2] * cx1) + \
            kp_obj[2] * (r[2, 0] - r[2, 2] * cx1) - \
            dz * (r[2, 0] - r[2, 2] * cx1)
        cy2 = \
            kp_obj[0] * (r[0, 1] - r[0, 2] * cy1) + \
            kp_obj[1] * (r[1, 1] - r[1, 2] * cy1) + \
            kp_obj[2] * (r[2, 1] - r[2, 2] * cy1) - \
            dz * (r[2, 1] - r[2, 2] * cy1)

        a = r[0, 0] - r[0, 2] * cx1
        b = r[1, 0] - r[1, 2] * cx1
        d = r[0, 1] - r[0, 2] * cy1
        e = r[1, 1] - r[1, 2] * cy1

        coefficients.append((
            a, b, cx2, d, e, cy2
        ))
    return coefficients


def construct_matrices_lsm(all_coefficients):
    A = np.ndarray([len(all_coefficients) * 2, 2])
    B = np.ndarray([len(all_coefficients) * 2, 1])
    for (coefficients, index) in zip(all_coefficients, range(len(all_coefficients))):
        A[index*2, 0] = coefficients[0]
        A[index*2, 1] = coefficients[1]
        B[index*2, 0] = coefficients[2]
        A[index*2+1, 0] = coefficients[3]
        A[index*2+1, 1] = coefficients[4]
        B[index*2+1, 0] = coefficients[5]

    return A, B


def calculate_optical_flow(current_kp, previous_kp):
    vectors = []
    for (kp1, kp2) in zip(current_kp, previous_kp):
        vec = np.ndarray([2, 1])
        vec[0] = kp1[0] - kp2[0]
        vec[1] = kp1[1] - kp2[1]
        vectors.append(vec.copy())

    return vectors


def calculate_of_parameters(vectors):
    a = []
    phi = []
    for v in vectors:
        a.append(math.sqrt(v[0]**2 + v[1]**2))
        phi.append(math.atan2(v[0], v[1]))

    return a, phi

def dense_optical_flow(
        img: np.ndarray,
        template_window_size: list,
        search_window_size: list,
        x_regions: int = 4,
        y_regions: int = 4,
):
    h, w = img.shape[:2]
    delta_x = w / (x_regions + 1)
    delta_y = h / (y_regions + 1)

    for x_index in range(x_regions):
        for y_index in range(y_regions):
            x_center, y_center = int(delta_x * (x_index + 1/2)), int(delta_y * (y_index + 1/2))

