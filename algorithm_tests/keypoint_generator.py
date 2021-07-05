from math import tan, floor, ceil

import numpy as np
from numpy import sign


def generate_keypoints(bottom_left_point: np.ndarray, top_left_point: np.ndarray,
                       top_right_point: np.ndarray, bottom_right_point: np.ndarray,
                       keypoint_amount: int = 10) -> list:
    """
    Generate specified amount of keypoints
    Generation is performed on two diagonals of specified points
    First line: bottom left point -> top right point
    Second line: top left point -> bottom right point
    Line parameters:
    1. angle
    2. bias
    :param bottom_left_point:
    :param top_left_point:
    :param top_right_point:
    :param bottom_right_point:
    :param keypoint_amount:
    :return: list of generated keypoints
    """
    # lambda for line calculation
    straight_line = lambda x, k, b: x*k + b

    # calculating line parameters
    k1 = (top_right_point[1] - bottom_left_point[1])/(top_right_point[0] - bottom_left_point[0])
    b1 = bottom_left_point[1]
    k2 = (bottom_right_point[1] - top_left_point[1])/(bottom_right_point[0] - top_left_point[0])
    b2 = top_left_point[1]

    # calculating generation parameters
    keypoints_per_line = ceil(keypoint_amount/2)
    delta_x_first_line = (top_right_point[0] - bottom_left_point[0])/(keypoints_per_line + 1)
    delta_x_second_line = (bottom_right_point[0] - top_left_point[0])/(keypoints_per_line + 1)

    result_keypoints = []
    for keypoint_index in range(keypoints_per_line):
        keypoint_x_first_line = bottom_left_point[0] + delta_x_first_line * (keypoint_index + 1)
        keypoint_x_second_line = top_left_point[0] + delta_x_second_line * (keypoint_index + 1)

        first_line_keypoint = np.ndarray([3, 1])
        second_line_keypoint = np.ndarray([3, 1])

        first_line_keypoint[0] = floor(keypoint_x_first_line)
        first_line_keypoint[1] = floor(straight_line(keypoint_x_first_line, k1, b1))
        first_line_keypoint[2] = 0

        second_line_keypoint[0] = floor(keypoint_x_second_line)
        second_line_keypoint[1] = floor(straight_line(keypoint_x_second_line, k2, b2))
        second_line_keypoint[2] = 0

        result_keypoints.append(first_line_keypoint)
        result_keypoints.append(second_line_keypoint)

    return result_keypoints


def transform_keypoints(initial_3d_keypoints: list, rotation_matrix: np.ndarray,
                        displacement_vector: np.ndarray) -> list:
    """
    Generate new 3d keypoints from initial 3d keypoints
    3D_2 = rotation_matrix*3D_1 + displacement_vector
    :param initial_3d_keypoints: initial 3d keypoints placement
    :param rotation_matrix: rotation matrix used to rotate keypoints
    :param displacement_vector: displacement vector to apply to keypoints
    :return:
    """
    transformed_keypoints = []
    for kp in initial_3d_keypoints:
        transformed_keypoints.append(rotation_matrix.dot(kp) + displacement_vector)

    return transformed_keypoints


def calculate_2d_from_3d(keypoints_3d: list, camera_3d_position: np.ndarray,
                         alpha: float, beta: float, resolution: list) -> list:
    """
    Algorithm of converting 3D to 2D:
    1. check if point is in camera perspective
    2. project a point into flat surface with Z = 0
    3. calculate coordinates in camera reference frame
    4. point's coordinate quantization
    :param keypoints_3d: list of keypoints in 3D space
    :param camera_3d_position: camera position in 3D space [X, Y, Z]
    :param alpha: camera's AOV along X axis
    :param beta: camera's AOV along Y axis
    :param resolution: camera's image resolution
    :return:
    """

    def to_camera_center(point: np.ndarray) -> np.ndarray:
        """
        Transformation to 3D camera frame of reference
        X_camera_3D = X_point - X_camera
        Y_camera_3D = Y_point - Y_camera
        Z_camera_3D = Z_camera - Z_point -> Z axis is inverted
        Supposing that camera is always above points
        If point is higher than camera -> point is not in camera perspective
        :param point: point in 3D space
        :return:
        """
        tmp = point - camera_3d_position
        tmp[2] = camera_3d_position[2] - point[2]
        return tmp

    def is_in_camera_perspective(point: np.ndarray) -> bool:
        """
        Check if camera can see the point
        Camera can see the point if:
        1. |x| < x(z), x(z) = h*tan(alpha/2)
        2. |y| < y(z), y(z) = h*tan(beta/2)
        3. z > 0
        :param point:
        :return:
        """
        max_delta_x = point[2]*tan(alpha/2)
        max_delta_y = point[2]*tan(beta/2)
        return (point[2] > 0) and (abs(point[0]) < max_delta_x) and (abs(point[1]) < max_delta_y)

    def project_point_to_surface(point: np.ndarray) -> np.ndarray:
        """
        Calculate projection coordinates of point on surface:
        X_proj = (H - z)*tan(x/z)*sign(x) + X_point
        Y_proj = (H - z)*tan(y/z)*sign(y) + Y_point
        Returns 2D coordinates on surface in camera's 2D reference frame
        :param point:
        :return:
        """
        tmp = np.ndarray([2, 1])
        tmp[0] = (camera_3d_position[2] - point[2]) * tan(point[0]/point[2]) * sign(point[0]) + point[0]
        tmp[1] = (camera_3d_position[2] - point[2]) * tan(point[1]/point[2]) * sign(point[1]) + point[1]
        return tmp

    def coordinate_quantization(point: np.ndarray) -> np.ndarray:
        """
        Quantize point coordinates to pixels
        Quantization is done in following way:
        X_surface -> X_pixels -> X_image
        1. Transform X_surface coordinates into pixel values
        X_pix = X / (meters in pixels)
        Y_pix = Y / (meters in pixels)
        2. Transform X_pixels into image coordinates
        X_im = - Y_q + resolution_x / 2
        Y_im = - X_q + resolution_y / 2
        Z axis in image coordinates faces earth
        Meters in pixels: distance / amount of pixels
        distance = H*tan(AOV_axis / 2)
        amount_of_pixels = resolution_axis / 2
        Y axis is inverted due to
        :param point: 2D point on surface
        :return:
        """
        tmp = np.ndarray([2, 1])
        pixel_resolution_x = camera_3d_position[2]*tan(alpha/2)/resolution[0]/2
        pixel_resolution_y = camera_3d_position[2]*tan(beta/2)/resolution[1]/2
        tmp[0] = floor(point[0] / pixel_resolution_x)
        tmp[1] = floor(point[1] / pixel_resolution_y)
        tmp[0], tmp[1] = - tmp[1] + resolution[0] / 2, -tmp[0] + resolution[1] / 2
        return tmp

    pixels_x_axis = 2*camera_3d_position[2]*tan(alpha/2)/resolution[0]
    pixels_y_axis = 2*camera_3d_position[2]*tan(beta/2)/resolution[1]

    mask = []
    keypoints_2d = []

    for kp in keypoints_3d:
        tmp_kp = to_camera_center(kp)
        if is_in_camera_perspective(tmp_kp):
            tmp_kp = project_point_to_surface(tmp_kp)
            tmp_kp = coordinate_quantization(tmp_kp)
            mask.append(True)
            keypoints_2d.append(tmp_kp)
        else:
            keypoints_2d.append([-1, -1])
            mask.append(False)

    return [keypoints_2d, mask]


def calculate_3d_from_2d(keypoints_2d: list, alpha: float, beta: float, height: float, resolution: list) -> list:
    pass
