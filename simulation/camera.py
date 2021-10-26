import math
import numpy as np

from simulation.rotation_matrix import calculate_rotation_matrix


class UFO:
    """
    UFO - class representing real moving object
    """
    def __init__(self, initial_position: np.ndarray, rotation_matrix: np.ndarray):
        self.position = initial_position.copy()
        self.rotation_matrix = rotation_matrix.copy()

    def rotate(self, rotation_matrix: np.ndarray):
        self.rotation_matrix = np.dot(rotation_matrix, self.rotation_matrix)

    def move(self, delta):
        self.position = self.position + delta


class Camera:

    def __init__(self, initial_position: np.ndarray, initial_rotation_matrix: np.ndarray,
                 f: float, fov: list, resolution: list):
        self.ufo = UFO(initial_position, initial_rotation_matrix)

        self.setup_angles = [
            0,
            math.pi,
            0
        ]

        self.ufo_r_to_camera = calculate_rotation_matrix(
            self.setup_angles[0],
            self.setup_angles[1],
            self.setup_angles[2]
        )

        self.position = self.ufo.position
        self.R = np.dot(self.ufo_r_to_camera, self.ufo.rotation_matrix)

        self.f = f
        self.fov = fov
        self.resolution = resolution
        matrix_size = [math.atan(fov[0] / 2 / math.pi) * f * 2, math.atan(fov[1] / 2 / math.pi) * f * 2]
        self.matrix_size = matrix_size
        self.px_mm = [matrix_size[0] / resolution[0], matrix_size[1] / resolution[1]]
        self.internal_matrix = np.eye(3)
        self.internal_matrix[0, 0] = self.f / self.px_mm[0]
        self.internal_matrix[1, 1] = self.f / self.px_mm[1]
        self.internal_matrix[0, 2] = self.resolution[0] / 2
        self.internal_matrix[1, 2] = self.resolution[1] / 2
        self.internal_matrix[2, 2] = 1

    def set_camera_position(self, new_camera_position: np.ndarray):
        self.position = new_camera_position.copy()
        self.ufo.position = new_camera_position.copy()

    def set_rotation(self, new_rotation_matrix: np.ndarray):
        self.ufo.rotation_matrix = new_rotation_matrix.copy()
        self.R = np.dot(self.ufo_r_to_camera, self.ufo.rotation_matrix)

    def move_camera(self, position_delta: np.ndarray):
        self.ufo.move(position_delta)
        self.position = self.ufo.position.copy()

    def rotate_camera(self, rotation_matrix: np.ndarray):
        self.ufo.rotate(rotation_matrix)
        self.R = np.dot(self.ufo_r_to_camera, self.ufo.rotation_matrix)

    def get_camera_position(self):
        return self.position

    def project_point_on_camera(self):
        return self.R


def calculate_camera_fov(camera: Camera) -> list:
    fov = [
        math.atan(camera.fov[0] / 2 / math.pi) * camera.position[2] * 2,
        math.atan(camera.fov[1] / 2 / math.pi) * camera.position[2] * 2,
    ]
    return fov
