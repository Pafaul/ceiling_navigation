import math
from math import tan, sqrt
import numpy as np

class Camera:

    def __init__(self, initial_position: np.ndarray, f: float, fov: list, resolution: list):
        self.position = initial_position
        self.i = np.zeros([3, 1])
        self.i[0] = 1
        self.j = np.zeros([3, 1])
        self.j[1] = 1
        self.k = np.zeros([3, 1])
        self.k[2] = 1
        self.R = np.eye(3)

        self.f = f
        self.fov = fov
        self.resolution = resolution
        matrix_size = [math.atan(fov[0] / 2 / math.pi) * f * 2, math.atan(fov[1] / 2 / math.pi) * f * 2]
        self.matrix_size = matrix_size
        self.px_mm = [resolution[0] / matrix_size[0], resolution[1] / matrix_size[1]]

    def set_camera_position(self, new_camera_position: np.ndarray):
        self.position = new_camera_position

    def move_camera(self, camera_position_delta: np.ndarray):
        self.position = self.position + camera_position_delta

    def rotate_camera(self, rotation_matrix: np.ndarray):
        self.i = np.dot(rotation_matrix, self.i)
        self.j = np.dot(rotation_matrix, self.j)
        self.k = np.dot(rotation_matrix, self.k)
        self.R = np.dot(rotation_matrix, self.R)

    def get_camera_position(self):
        pass

    def project_point_on_camera(self):
        pass

    def get_point_coordinates_on_image(self):
        pass
