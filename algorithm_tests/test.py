import cv2
import numpy as np
from math import pi

from algorithm_tests.camera import Camera
from algorithm_tests.keypoint_generator import calculate_2d_from_3d, transform_keypoints, generate_keypoints_equal
from algorithm_tests.utils import calculate_rotation_matrix
from algorithm_tests.optical_flow_visualization import visualize_optical_flow, visualize_not_moving_coordinates


def main():
    rad_to_deg = lambda x: x*pi/180
    camera_position = np.ndarray([3, 1])
    camera_position[0] = 0
    camera_position[1] = 0
    camera_position[2] = 100
    camera = Camera([600, 600], 1, [rad_to_deg(60), rad_to_deg(60)], camera_position)

    bottom_left_generation_corner = [-300, -300]
    height, width = 600, 600
    kp1 = np.ndarray([3, 1])
    kp1[0] = bottom_left_generation_corner[0]
    kp1[1] = bottom_left_generation_corner[1]
    kp1[2] = 0
    kp2 = np.ndarray([3, 1])
    kp2[0] = bottom_left_generation_corner[0]
    kp2[1] = bottom_left_generation_corner[1] + height
    kp2[2] = 0
    kp3 = np.ndarray([3, 1])
    kp3[0] = bottom_left_generation_corner[0] + width
    kp3[1] = bottom_left_generation_corner[1] + height
    kp3[2] = 0
    kp4 = np.ndarray([3, 1])
    kp4[0] = bottom_left_generation_corner[0] + width
    kp4[1] = bottom_left_generation_corner[1]
    kp4[2] = 0

    generated_keypoints = generate_keypoints_equal(kp1, kp3, x_keypoint_amount=15)

    displacement_vector = np.ndarray([3, 1])
    displacement_vector[0] = 100
    displacement_vector[1] = 100
    displacement_vector[2] = 0

    rotation_matrix = calculate_rotation_matrix(rad_to_deg(0), rad_to_deg(0), rad_to_deg(45))

    camera.position += displacement_vector
    camera_fov = camera.get_camera_fov_in_real_coordinates(rotation_matrix)

    kp_2d, mask_1 = calculate_2d_from_3d(generated_keypoints, camera)

    transformed_kp = transform_keypoints(generated_keypoints, rotation_matrix, displacement_vector)
    transformed_kp_2d, mask_2 = calculate_2d_from_3d(transformed_kp, camera)
    kp_2d_1 = []
    kp_2d_2 = []

    for kp_first_image, kp_second_image, is_visible_1, is_visible_2 in zip(kp_2d, transformed_kp_2d, mask_1, mask_2):
        if is_visible_1 and is_visible_2:
            kp_2d_1.append(kp_first_image)
            kp_2d_2.append(kp_second_image)

    kps_1 = np.int32(kp_2d_1)
    kps_2 = np.int32(kp_2d_2)

    camera_matrix = np.array([
        [2.20577589e+03, 0.00000000e+00, 3.01560236e+02],
        [0.00000000e+00, 2.20577589e+03, 3.01560236e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    img_not_moving = visualize_not_moving_coordinates(generated_keypoints, transformed_kp, camera_fov, [kp1*2, kp3*2])
    cv2.imshow('stationary', img_not_moving)
    cv2.waitKey(0)

    image = np.ones(camera.resolution, dtype=np.uint8)*128

    # print(rotation_matrix)
    E, mask = cv2.findEssentialMat(kps_1, kps_2, camera_matrix)
    retval, R, t, mask = cv2.recoverPose(E, kps_1, kps_2, camera_matrix, mask=mask)
    # print(R)

    image = visualize_optical_flow(kps_1, kps_2, image)
    cv2.imshow('test', image)
    cv2.waitKey(0)

    # for a, b in zip(kp_2d, transformed_kp_2d):
    #     print(a)

    # X, emissions, error_EV, error_DP = seq_lsm(kp_2d_1, kp_2d_2, calculate_rotation_matrix(rad_to_deg(0), rad_to_deg(0), rad_to_deg(-10)))
    # print(X)

if __name__ == '__main__':
    main()