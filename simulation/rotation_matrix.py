import math

import numpy as np


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
