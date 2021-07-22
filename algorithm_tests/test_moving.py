import cv2
import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import sin

from algorithm_tests.camera import Camera
from algorithm_tests.keypoint_generator import generate_keypoints_equal
from algorithm_tests.utils import calculate_rotation_matrix, calculate_rotation_matrix_from_euler_angles
from algorithm_tests.optical_flow_visualization import visualize_not_moving_coordinates, \
    visualize_keypoints, visualize_camera_fov, visualize_keypoints_without_transformation, visualize_keypoints_deltas
from lsm.least_square_method import seq_lsm


def deg_to_rad(x: float) -> float:
    return x*pi/180


def rad_to_deg(x: float) -> float:
    return x*180/pi


def calculate_angles_from_rotation_matrix(R: np.ndarray) -> list:
    return [
        rad_to_deg(atan(R[2][1] / R[2][2])),
        rad_to_deg(atan(-R[2][0] / ((R[2][1] ** 2 + R[2][2] ** 2) ** 0.5))),
        rad_to_deg(atan(R[1][0] / R[0][0]))
    ]


def main():
    camera_position = np.ndarray([3, 1])
    camera_position[0] = 0
    camera_position[1] = 0
    camera_position[2] = 300
    camera = Camera([600, 600], 1, [deg_to_rad(60), deg_to_rad(60)], camera_position.copy())

    bottom_left_generation_corner = [-1200, -1200]
    height, width = 2400, 2400
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
        [1000, 0.00000000e+00, 300],
        [0.00000000e+00, 1000, 300],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    generated_keypoints = generate_keypoints_equal(kp1, kp3, x_keypoint_amount=60, y_keypoint_amount=60)

    zero_rotation_matrix = calculate_rotation_matrix(0, 0, 0)
    real_angles = []
    calculated_angles_1 = []
    calculated_angles_2 = []

    keypoint_previous = []
    mask_previous = []
    keypoint_current = []
    mask_current = []

    x = np.linspace(0, pi*6, 100, endpoint=False)
    y_1 = sin(x[:-1])*10
    y_2 = sin(x[1:])*10
    displacement_vector = np.ndarray([3, 1])
    displacement_vector[0] = 0
    displacement_vector[1] = 0
    displacement_vector[2] = 0
    result_rotation_matrix = np.eye(3)
    result_angles = []
    [keypoint_previous, mask_previous] = camera.calculate_keypoint_projections_to_plane(
        generated_keypoints,
        zero_rotation_matrix
    )

    total_angle_real = [0]
    total_angle_calc = [0]
    angle_index = 0

    for angle_1, angle_2 in zip(y_1, y_2):
        angle = -(angle_2 - angle_1)
        total_angle_real.append(total_angle_real[-1] + angle)
        camera.position += displacement_vector
        camera.update_parameters()

        rotation_matrix = calculate_rotation_matrix_from_euler_angles(deg_to_rad(0), deg_to_rad(angle), deg_to_rad(0))
        result_rotation_matrix = np.matmul(result_rotation_matrix, rotation_matrix)

        image_camera = np.ones(camera.resolution, dtype=np.uint8) * 128

        [keypoint_current, mask_current] = camera.calculate_keypoint_projections_to_plane(
            generated_keypoints,
            result_rotation_matrix
        )

        kp_2d_1 = []
        kp_2d_2 = []

        for kp_first_image, kp_second_image, is_visible_1, is_visible_2 in zip(keypoint_current, keypoint_previous,
                                                                               mask_current, mask_previous):
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

        image_camera = visualize_keypoints_without_transformation(image_camera, kp_2d_1, cv2.MARKER_DIAMOND)
        cv2.imshow('camera view 1', image_camera)

        image_camera = visualize_keypoints_without_transformation(image_camera, kp_2d_2, cv2.MARKER_CROSS)
        cv2.imshow('camera view 2', image_camera)

        image_camera = visualize_keypoints_deltas(image_camera, kp_2d_1, kp_2d_2)
        cv2.imshow('camera view 3', image_camera)

        cv2.waitKey(1)

        E, mask = cv2.findEssentialMat(kps_2, kps_1, camera_matrix, method=cv2.LMEDS, threshold=2)
        R1, R2, t = cv2.decomposeEssentialMat(E)
        # retval, R, t, mask = cv2.recoverPose(E, kps_1, kps_2, camera_matrix, mask=mask)

        keypoint_previous = keypoint_current.copy()
        mask_previous = mask_current.copy()
        real_angles.append(angle)
        calculated_angles_1.append(calculate_angles_from_rotation_matrix(R1))
        calculated_angles_2.append(calculate_angles_from_rotation_matrix(R2))
        if abs(calculated_angles_1[-1][angle_index] - angle) < abs(calculated_angles_2[-1][angle_index] - angle):
            result_angles.append(calculated_angles_1[-1][angle_index])
        else:
            result_angles.append(calculated_angles_2[-1][angle_index])

        total_angle_calc.append(total_angle_calc[-1] + result_angles[-1])

    plt.plot(real_angles, 'b')
    blue_patch = mpatches.Patch(color='blue', label='Действительный угол')
    # plt.plot([angle[0] for angle in calculated_angles_1], 'r')
    # plt.plot([angle[1] for angle in calculated_angles_1], 'g')
    plt.plot(result_angles, 'g')
    green_label = mpatches.Patch(color='green', label='Рассчитанный угол')
    #plt.plot([angle[2] for angle in calculated_angles_1], 'y')
    plt.legend(handles=[blue_patch, green_label], loc='upper right')
    plt.show()

    plt.plot(total_angle_real, 'b')
    plt.plot(total_angle_calc, 'g')
    blue_patch = mpatches.Patch(color='blue', label='Действительный угол')
    green_label = mpatches.Patch(color='green', label='Рассчитанный угол')
    plt.legend(handles=[blue_patch, green_label], loc='upper right')
    plt.show()

    plt.plot([x - y for x, y in zip(total_angle_real, total_angle_calc)], 'g')
    plt.show()

    print(total_angle_real[-1] - total_angle_calc[-1])

    # plt.plot(real_angles, 'b')
    # # plt.plot([angle[0] for angle in calculated_angles_2], 'r')
    # # plt.plot([angle[1] for angle in calculated_angles_2], 'g')
    #
    # plt.show()


if __name__ == '__main__':
    main()
