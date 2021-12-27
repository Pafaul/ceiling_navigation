import numpy as np
from cv2 import cv2
import math


def calculate_rotation_matrix(wx: float, wy: float, wz: float) -> np.ndarray:
    r_x = np.eye(3)
    r_y = np.eye(3)
    r_z = np.eye(3)

    r_z[0][0] = math.cos(wz)
    r_z[0][1] = -math.sin(wz)
    r_z[1][0] = math.sin(wz)
    r_z[1][1] = math.cos(wz)
    r_z[2][2] = 1

    r_y[0][0] = math.cos(wy)
    r_y[0][2] = math.sin(wy)
    r_y[1][1] = 1
    r_y[2][0] = -math.sin(wy)
    r_y[2][2] = math.cos(wy)

    r_x[0][0] = 1
    r_x[1][1] = math.cos(wx)
    r_x[1][2] = -math.sin(wx)
    r_x[2][1] = math.sin(wx)
    r_x[2][2] = math.cos(wx)

    rotation_matrix = r_z
    rotation_matrix = np.dot(r_y, rotation_matrix)
    rotation_matrix = np.dot(r_x, rotation_matrix)
    return rotation_matrix


def calculate_angles(R: np.ndarray) -> list:
    def rad_to_deg(x):
        return x * 180 / math.pi

    return [
        rad_to_deg(math.atan2(R[2, 1] , R[2, 2])),
        rad_to_deg(-math.asin(R[2, 0])),
        rad_to_deg(math.atan2(R[1, 0] , R[0, 0]))
    ]


def get_abs_diff_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    r = abs(R1 - R2)
    return sum(sum(r))


def fix_r(r: np.ndarray):
    res_r = np.zeros([3, 3])
    eps = 0
    if abs(1 - r[0][0]) < eps:
        res_r = calculate_rotation_matrix(0, 0, math.atan2(r[1, 0], r[0, 0]))
    elif abs(-1 - r[0, 0]) < eps:
        res_r = calculate_rotation_matrix(0, 0, math.atan2(r[1, 0], r[0, 0]))
    else:
        res_r = r.copy()
    return res_r


def get_correct_r(r1, r2):
    angles1 = calculate_angles(r1)
    angles2 = calculate_angles(r2)
    count = sum([1 if abs(angle_1) < abs(angle_2) else 0 for (angle_1, angle_2) in zip(angles1, angles2)])
    if count >= 2:
        return r1
    else:
        return r2


def filter_optical_flow(of: list):
    a, phi = calculate_of_parameters(of)

    mean_a = np.mean(a)
    disp_a = np.std(a)

    mean_phi = np.mean(phi)
    disp_phi = np.std(phi)

    mask_a = [True if abs(cur_a - mean_a) < 2*disp_a else False for cur_a in a]
    mask_phi = [True if abs(cur_phi - mean_phi) < 2*disp_phi else False for cur_phi in phi]
    res_mask = [m_a and m_phi for (m_a, m_phi) in zip(mask_a, mask_phi)]
    res_of = []
    for i in range(len(res_mask)):
        if res_mask[i]:
            res_of.append(of[i])

    return res_mask, res_of


def calculate_obj_rotation_matrix(
        previous_kp: np.int32,
        current_kp: np.int32,
        camera_internal_matrix: np.ndarray,
        rotation_matrix: np.ndarray
):
    E, mask = cv2.findEssentialMat(previous_kp, current_kp, camera_internal_matrix, method=cv2.LMEDS, threshold=0.1)
    R1, R2, t = cv2.decomposeEssentialMat(E)
    R1 = fix_r(R1)
    R2 = fix_r(R2)
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


def px_to_obj_2(
        keypoints: np.int32,
        r: np.ndarray,
        position: np.ndarray,
        camera_matrix: np.ndarray,
        h: float
):
    u0, v0 = camera_matrix[0, 2], camera_matrix[1, 2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    keypoints_obj = []
    for kp in list(keypoints):
        keypoint_position = np.zeros([3, 1])
        keypoint_position[2] = h
        nom_x = r[0, 0] * (kp[0] - u0) + r[0, 1] * (kp[1] - v0) - r[0, 2] * fx
        nom_y = r[1, 0] * (kp[0] - u0) + r[1, 1] * (kp[1] - v0) - r[1, 2] * fy
        denom_x = r[2, 0] * (kp[0] - u0) + r[2, 1] * (kp[1] - v0) - r[2, 2] * fx
        denom_y = r[2, 0] * (kp[0] - u0) + r[2, 1] * (kp[1] - v0) - r[2, 2] * fy
        keypoint_position[0] = keypoint_position[2] * nom_x / denom_x
        keypoint_position[1] = keypoint_position[2] * nom_y / denom_y
        keypoints_obj.append(keypoint_position.copy())

    return keypoints_obj


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


def prepare_data_for_lsm_2(
    keypoints: np.int32,
    keypoints_obj: list,
    r: np.ndarray,
    camera_matrix: np.ndarray,
    h: float
):
    coefficients = []
    u0, v0 = camera_matrix[0, 2], camera_matrix[1, 2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    for index in range(len(keypoints_obj)):
        cx1 = (u0 - keypoints[index][0])
        cy1 = (v0 - keypoints[index][1])
        a1 = cx1 * r[0, 2] - fx * r[0, 0]
        a2 = cx1 * r[1, 2] - fx * r[1, 0]
        a3 = cx1 * r[2, 2] - fx * r[2, 0]
        b1 = cy1 * r[0, 2] - fy * r[0, 1]
        b2 = cy1 * r[1, 2] - fy * r[1, 1]
        b3 = cy1 * r[2, 2] - fy * r[2, 1]
        cx2 = keypoints_obj[index][0] * a1 + keypoints_obj[index][1] * a2 + keypoints_obj[index][2] * a3
        cy2 = keypoints_obj[index][0] * b1 + keypoints_obj[index][1] * b2 + keypoints_obj[index][2] * b3
        coefficients.append(
            [a1, a2, a3, cx2, b1, b2, b3, cy2]
        )

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


def construct_matrices_lsm_2(coefficients: list):
    A = np.ndarray([len(coefficients) * 2, 3])
    B = np.ndarray([len(coefficients) * 2, 1])
    for (c, i) in zip(coefficients, range(len(coefficients))):
        A[i * 2, 0] = c[0]
        A[i * 2, 1] = c[1]
        A[i * 2, 2] = c[2]
        B[i * 2, 0] = c[3]
        A[i * 2 + 1, 0] = c[4]
        A[i * 2 + 1, 1] = c[5]
        A[i * 2 + 1, 2] = c[6]
        B[i * 2 + 1, 0] = c[7]

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

