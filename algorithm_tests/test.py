import numpy as np
from math import pi
from algorithm_tests.keypoint_generator import calculate_2d_from_3d, transform_keypoints, generate_keypoints
from algorithm_tests.utils import calculate_rotation_matrix_from_euler_angles


def main():
    rad_to_deg = lambda x: x*pi/180
    kp1 = np.ndarray([3, 1])
    kp1[0] = 0; kp1[1] = 0; kp1[2] = 0
    kp2 = np.ndarray([3, 1])
    kp2[0] = 0; kp2[1] = 100; kp2[2] = 0
    kp3 = np.ndarray([3, 1])
    kp3[0] = 100; kp3[1] = 100; kp3[2] = 0
    kp4 = np.ndarray([3, 1])
    kp4[0] = 100; kp4[1] = 0; kp4[2] = 0

    generated_keypoints = generate_keypoints(kp1, kp2, kp3, kp4)

    displacement_vector = np.ndarray([3, 1])
    displacement_vector[0] = 0
    displacement_vector[1] = 0
    displacement_vector[2] = 0

    rotation_matrix = calculate_rotation_matrix_from_euler_angles(rad_to_deg(0), rad_to_deg(0), rad_to_deg(90))
    camera_position = np.ndarray([3, 1])
    camera_position[0] = 0
    camera_position[1] = 0
    camera_position[2] = 1000

    kp_2d = calculate_2d_from_3d(generated_keypoints, camera_position, rad_to_deg(60), rad_to_deg(60), [600, 600])

    transformed_kp = transform_keypoints(generated_keypoints, rotation_matrix, displacement_vector)


if __name__ == '__main__':
    main()