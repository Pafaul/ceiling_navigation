import cv2
import numpy as np
from math import pi, atan2, atan
import matplotlib.pyplot as plt

from algorithm_tests.camera import Camera
from algorithm_tests.keypoint_generator import generate_keypoints_equal, generate_keypoints
from algorithm_tests.utils import calculate_rotation_matrix, calculate_rotation_matrix_from_euler_angles
from algorithm_tests.optical_flow_visualization import visualize_not_moving_coordinates, \
    visualize_keypoints, visualize_camera_fov, visualize_keypoints_without_transformation, visualize_keypoints_deltas
from lsm.least_square_method import seq_lsm


def main():
    deg_to_rad = lambda x: x * pi / 180
    rad_to_deg = lambda x: x * 180 / pi

    camera_position = np.ndarray([3, 1])
    camera_position[0] = 0
    camera_position[1] = 0
    camera_position[2] = 300
    camera = Camera([600, 600], 1, [deg_to_rad(60), deg_to_rad(60)], camera_position.copy())

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

    boundaries = [kp1 * 2, kp3 * 2]

    camera_matrix = np.array([
        [2000, 0.00000000e+00, 300],
        [0.00000000e+00, 2000, 300],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    generated_keypoints = generate_keypoints_equal(kp1, kp3, x_keypoint_amount=16, y_keypoint_amount=20)
    # generated_keypoints = generate_keypoints(kp1, kp2, kp3, kp4, keypoint_amount=150)
    # generated_keypoints = [np.zeros([3, 1])]
    zero_rotation_matrix = calculate_rotation_matrix(0, 0, 0)
    real_angles = []
    calculated_angles = []

    for angle in np.linspace(0, 30, 100, endpoint=False):
        camera.position = camera_position.copy()
        displacement_vector = np.ndarray([3, 1])
        displacement_vector[0] = 0
        displacement_vector[1] = 0
        displacement_vector[2] = 10

        rotation_matrix = calculate_rotation_matrix_from_euler_angles(deg_to_rad(0), deg_to_rad(angle), deg_to_rad(0))

        camera.update_parameters()
        img_not_moving = visualize_not_moving_coordinates(boundaries)
        img_not_moving = visualize_keypoints(img_not_moving, generated_keypoints, boundaries, cv2.MARKER_CROSS)
        img_not_moving = visualize_camera_fov(img_not_moving,
                                              camera.get_camera_fov_in_real_coordinates(zero_rotation_matrix), boundaries, (0,))
        camera.position += displacement_vector
        camera.update_parameters()
        img_not_moving = visualize_camera_fov(img_not_moving,
                                              camera.get_camera_fov_in_real_coordinates(rotation_matrix), boundaries, (255,))

        cv2.imshow('stationary', img_not_moving)

        camera.position = camera_position.copy()
        camera.update_parameters()
        image_camera = np.ones(camera.resolution, dtype=np.uint8) * 128
        [kp_first_image, first_image_mask] = camera.calculate_keypoint_projections_to_plane(
            generated_keypoints,
            zero_rotation_matrix
        )

        camera.position += displacement_vector
        camera.update_parameters()
        [kp_second_image, second_image_mask] = camera.calculate_keypoint_projections_to_plane(
            generated_keypoints,
            rotation_matrix
        )

        kp_2d_1 = []
        kp_2d_2 = []

        for kp_first_image, kp_second_image, is_visible_1, is_visible_2 in zip(kp_first_image, kp_second_image,
                                                                               first_image_mask, second_image_mask):
            if is_visible_1 and is_visible_2:
                kp_2d_1.append(kp_first_image.copy())
                kp_2d_2.append(kp_second_image.copy())

        kps_1 = np.int32(kp_2d_1)
        kps_2 = np.int32(kp_2d_2)

        f = open('kp1.txt', 'w')
        print(kp_2d_1, file=f)
        f.close()
        f = open('kp2.txt', 'w')
        print(kp_2d_2, file=f)
        f.close()

        image_camera = visualize_keypoints_without_transformation(image_camera, kp_2d_1, cv2.MARKER_CROSS)
        cv2.imshow('camera view 1', image_camera)

        image_camera = visualize_keypoints_without_transformation(image_camera, kp_2d_2, cv2.MARKER_DIAMOND)
        cv2.imshow('camera view 2', image_camera)

        image_camera = visualize_keypoints_deltas(image_camera, kp_2d_1, kp_2d_2)
        cv2.imshow('camera view 3', image_camera)

        # image_camera = visualize_optical_flow(image_camera, kps_1, kps_2)

        cv2.waitKey(50)

        E, mask = cv2.findEssentialMat(kps_1, kps_2, camera_matrix, method=cv2.RANSAC, threshold=5)
        retval, R, t, mask = cv2.recoverPose(E, kps_1, kps_2, camera_matrix, mask=mask)
        print(rotation_matrix)
        print(R)
        angles = [
            atan(R[2][1] / R[2][2]),
            atan(-R[2][0] / ((R[2][1] ** 2 + R[2][2] ** 2) ** 0.5)),
            atan(R[1][0] / R[0][0])
        ]
        real_angles.append(angle)
        calculated_angles.append([rad_to_deg(angles[0]), rad_to_deg(angles[1]), rad_to_deg(angles[2])])

    plt.plot(real_angles, 'b')
    plt.plot([angle[0] for angle in calculated_angles], 'r')
    plt.plot([angle[1] for angle in calculated_angles], 'g')
    plt.plot([angle[2] for angle in calculated_angles], 'y')
    plt.show()


if __name__ == '__main__':
    main()
