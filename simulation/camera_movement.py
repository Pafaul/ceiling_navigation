import math

import numpy as np

from simulation.camera import Camera
from simulation.rotation_matrix import calculate_rotation_matrix


class BasicCameraMovement:
    def move_camera(self, camera: Camera):
        pass

    def copy(self):
        pass


class LinearMovement(BasicCameraMovement):
    def __init__(self, start_point: np.array, finish_point: np.array, initial_angle: list, finish_angle: list, stop_points: int):
        self.moving_deltas = []
        delta_x = (finish_point[0] - start_point[0]) / stop_points
        delta_y = (finish_point[1] - start_point[1]) / stop_points
        delta_z = (finish_point[2] - start_point[2]) / stop_points

        self.rotation_deltas = []
        delta_wx = (finish_angle[0] - initial_angle[0]) * math.pi / 180 / stop_points
        delta_wy = (finish_angle[1] - initial_angle[1]) * math.pi / 180 / stop_points
        delta_wz = (finish_angle[2] - initial_angle[2]) * math.pi / 180 / stop_points

        self.movement_points = stop_points

        for i in range(stop_points):
            delta = np.ndarray([3, 1])
            delta[0] = delta_x
            delta[1] = delta_y
            delta[2] = delta_z
            self.moving_deltas.append(delta)
            self.rotation_deltas.append([delta_wx, delta_wy, delta_wz])

    def move_camera(self, camera: Camera):
        for (delta_movement, delta_rotation) in zip(self.moving_deltas, self.rotation_deltas):
            rotation_matrix = calculate_rotation_matrix(delta_rotation[0], delta_rotation[1], delta_rotation[2])
            camera.rotate_camera(rotation_matrix)
            yield camera.move_camera(delta_movement)


class SinMovement(BasicCameraMovement):
    def __init__(self, amplitude_x: np.array, amplitude_w: np.array, sin_max: float, stop_points: int):
        t = np.linspace(0, sin_max, stop_points, endpoint=False)
        self.sin_max = sin_max
        self.first_points = np.sin(t[:-1])
        self.second_points = np.sin(t[1:])
        self.amplitude_x = amplitude_x.copy()
        self.amplitude_w = amplitude_w.copy()
        self.movement_points = stop_points

    def move_camera(self, camera: Camera):
        for (t1, t2) in zip(self.first_points, self.second_points):
            delta = t2 - t1
            delta_angles = self.amplitude_w * delta
            delta_position = self.amplitude_x * delta
            rotation_matrix = calculate_rotation_matrix(delta_angles[0], delta_angles[1], delta_angles[2])
            camera.rotate_camera(rotation_matrix)
            yield camera.move_camera(delta_position)

    def copy(self):
        return SinMovement(
            self.amplitude_x,
            self.amplitude_w,
            self.sin_max,
            self.movement_points
        )