from math import cos, sin
import numpy as np


def calculate_rotation_matrix_from_euler_angles(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Calculate rotation matrix from euler angles
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
