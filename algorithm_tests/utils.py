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
