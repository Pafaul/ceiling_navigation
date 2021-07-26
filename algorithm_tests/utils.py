from math import cos, sin
import numpy as np


def calculate_rotation_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Calculate rotation matrix
    :param alpha: Angle between X axis
    :param beta: Angle between Y axis
    :param gamma: Angle between Z axis
    :return: Result rotation matrix
    """
    rotation_matrix = np.ndarray([3, 3])
    rotation_matrix[0, 0] =  cos(alpha)*cos(gamma) - sin(alpha)*cos(beta)*sin(gamma)
    rotation_matrix[0, 1] = -cos(alpha)*sin(gamma) - sin(alpha)*cos(beta)*cos(gamma)
    rotation_matrix[0, 2] =  sin(alpha)*sin(beta)
    rotation_matrix[1, 0] =  sin(alpha)*cos(gamma) + cos(alpha)*cos(beta)*sin(gamma)
    rotation_matrix[1, 1] = -sin(alpha)*sin(gamma) + cos(alpha)*cos(beta)*cos(gamma)
    rotation_matrix[1, 2] = -cos(alpha)*sin(beta)
    rotation_matrix[2, 0] =  sin(beta)*sin(gamma)
    rotation_matrix[2, 1] =  sin(beta)*cos(gamma)
    rotation_matrix[2, 2] =  cos(beta)

    return rotation_matrix


def calculate_rotation_matrix_from_euler_angles(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Calculate rotation matrix from euler angles
    :param yaw: about Z axis
    :param pitch: about Y axis
    :param roll: about X axis
    :return:
    """
    rotation_matrix = np.ndarray([3, 3])
    alpha = yaw
    beta = pitch
    gamma = roll
    rotation_matrix[0, 0] = cos(alpha)*cos(beta)
    rotation_matrix[0, 1] = cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma)
    rotation_matrix[0, 2] = cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma)
    rotation_matrix[1, 0] = sin(alpha)*cos(beta)
    rotation_matrix[1, 1] = sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma)
    rotation_matrix[1, 2] = sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma)
    rotation_matrix[2, 0] = -sin(beta)
    rotation_matrix[2, 1] = cos(beta)*sin(gamma)
    rotation_matrix[2, 2] = cos(beta)*cos(gamma)
    return rotation_matrix


def calculate_rotation_matrix_from_pdf(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    R1 = np.eye(3)
    R2 = np.eye(3)
    R3 = np.eye(3)
    c1, s1 = cos(angle_x), sin(angle_x)
    c2, s2 = cos(angle_y), sin(angle_y)
    c3, s3 = cos(angle_z), sin(angle_z)
    R1[1, 1], R1[1, 2] = c1, s1
    R1[2, 1], R1[2, 2] =-s1, c1
    R2[0, 0], R2[0, 2] = c2,-s2
    R2[2, 0], R2[2, 2] = s2, c2
    R3[0, 0], R3[0, 1] = c3, s3
    R3[1, 0], R3[1, 1] =-s3, c3
    return np.matmul(np.matmul(R1, R2), R3)


def calculate_rotation_matrix_euler_angles(pitch: float, roll: float, yaw: float) -> np.ndarray:
    B = np.eye(3)
    B[1, 1], B[1, 2] = cos(roll), sin(roll)
    B[2, 1], B[2, 2] = -sin(roll), cos(roll)
    C = np.eye(3)
    C[0, 0], C[0, 2] = cos(pitch), -sin(pitch)
    C[2, 0], C[2, 2] = sin(pitch), cos(pitch)
    D = np.eye(3)
    D[0, 0], D[0, 1] = cos(yaw), sin(yaw)
    D[1, 0], D[1, 1] = -sin(yaw), cos(yaw)
    return np.matmul(np.matmul(B, C), D)