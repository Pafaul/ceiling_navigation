import cv2
import numpy as np
from math import pi, atan, atan2, cos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import sin

from algorithm_tests.camera import Camera
from algorithm_tests.keypoint_generator import generate_keypoints_equal
from algorithm_tests.utils import calculate_rotation_matrix, calculate_rotation_matrix_from_euler_angles, \
    calculate_rotation_matrix_from_pdf, calculate_rotation_matrix_euler_angles
from algorithm_tests.optical_flow_visualization import visualize_not_moving_coordinates, \
    visualize_keypoints, visualize_camera_fov, visualize_keypoints_without_transformation, visualize_keypoints_deltas
from lsm.least_square_method import seq_lsm


def deg_to_rad(x: float) -> float:
    return x*pi/180


def rad_to_deg(x: float) -> float:
    return x*180/pi


# def calculate_angles_from_rotation_matrix(R: np.ndarray) -> list:
#     angle_x = atan2(R[1,2], R[2, 2])
#     c2 = (R[0, 0]**2 + R[0, 1]**2)**0.5
#     angle_y = atan2(-R[0, 2], c2)
#     s1 = sin(angle_x)
#     c1 = cos(angle_x)
#     angle_z = atan2(s1*R[2, 0] - c1*R[1, 0], c1*R[1, 1] - s1*R[2, 1])
#     return [
#         rad_to_deg(angle_y),
#         rad_to_deg(angle_x),
#         rad_to_deg(angle_z)
#     ]

def calculate_angles(R: np.ndarray) -> list:
    return [
        rad_to_deg(atan(-R[2][0] / ((R[2][1] ** 2 + R[2][2] ** 2) ** 0.5))),
        rad_to_deg(atan(R[2][1] / R[2][2])),
        rad_to_deg(atan(R[1][0] / R[0][0]))
    ]


def calculate_angles_from_rotation_matrix(R: np.ndarray) -> list:
    return [
        rad_to_deg(atan(R[2][1] / R[2][2])),
        rad_to_deg(atan(-R[2][0] / ((R[2][1] ** 2 + R[2][2] ** 2) ** 0.5))),
        rad_to_deg(atan(R[1][0] / R[0][0]))
    ]


def get_abs_diff(angles_1: list, angles_2: list) -> float:
    return sum([(a1 - a2)**2 for a1, a2 in zip(angles_1, angles_2)])


def l2r(R: np.ndarray) -> np.ndarray:
    rotation_matrix = R.copy()
    rotation_matrix[0, 2] *= -1
    rotation_matrix[1, 2] *= -1
    rotation_matrix[2, 0] *= -1
    rotation_matrix[2, 1] *= -1
    return rotation_matrix


def main():
    camera_position = np.ndarray([3, 1])
    camera_position[0] = 0
    camera_position[1] = 0
    camera_position[2] = 300
    camera = Camera([600, 600], 1, [deg_to_rad(60), deg_to_rad(60)], camera_position.copy())

    bottom_left_generation_corner = [-800, -800]
    height, width = 1600, 1600
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

    generated_keypoints = generate_keypoints_equal(kp1, kp3, x_keypoint_amount=30, y_keypoint_amount=30)

    zero_rotation_matrix = calculate_rotation_matrix(0, 0, 0)
    real_angles = []
    calculated_angles_1 = []
    calculated_angles_2 = []

    keypoint_previous = []
    mask_previous = []
    keypoint_current = []
    mask_current = []

    x = np.linspace(0, pi*6, 50, endpoint=False)
    y_1 = sin(x[:-1])*10
    y_2 = sin(x[1:])*10
    displacement_vector = np.ndarray([3, 1])
    displacement_vector[0] = 0
    displacement_vector[1] = 0
    displacement_vector[2] = 0
    result_rotation_matrix = np.eye(3)
    calculated_rotation_matrix = np.eye(3)
    tmp_rotation_matrix_1 = np.eye(3)
    tmp_rotation_matrix_2 = np.eye(3)
    result_angles = []
    [keypoint_previous, mask_previous] = camera.calculate_keypoint_projections_to_plane(
        generated_keypoints,
        zero_rotation_matrix
    )

    angle_index = 2

    for angle_1, angle_2 in zip(y_1, y_2):
        angle = (angle_2 - angle_1)
        camera.position += displacement_vector
        camera.update_parameters()

        rotation_matrix = calculate_rotation_matrix_from_euler_angles(deg_to_rad(angle), deg_to_rad(0), deg_to_rad(0))
        result_rotation_matrix = np.matmul(rotation_matrix, result_rotation_matrix)

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
        # R1 *= -1
        # R2 *= -1
        # R1 = l2r(R1)
        # R2 = l2r(R2)
        # retval, R, t, mask = cv2.recoverPose(E, kps_1, kps_2, camera_matrix, mask=mask)

        tmp_rotation_matrix_1 = np.matmul(R1, calculated_rotation_matrix)
        tmp_rotation_matrix_2 = np.matmul(R2, calculated_rotation_matrix)

        angles_current = calculate_angles(calculated_rotation_matrix)
        angles_1 = calculate_angles(tmp_rotation_matrix_1)
        angles_2 = calculate_angles(tmp_rotation_matrix_2)

        eps_1 = get_abs_diff(angles_current, angles_1)
        eps_2 = get_abs_diff(angles_current, angles_2)

        if eps_1 < eps_2:
            calculated_rotation_matrix = np.matmul(R1, calculated_rotation_matrix)
        else:
            calculated_rotation_matrix = np.matmul(R2, calculated_rotation_matrix)

        keypoint_previous = keypoint_current.copy()
        mask_previous = mask_current.copy()
        result_angles.append(calculate_angles(calculated_rotation_matrix))

        real_angles.append(calculate_angles_from_rotation_matrix(result_rotation_matrix))
        # calculated_angles_1.append(calculate_angles_from_rotation_matrix(R1))
        # calculated_angles_2.append(calculate_angles_from_rotation_matrix(R2))
        # if abs(calculated_angles_1[-1][angle_index] - angle) < abs(calculated_angles_2[-1][angle_index] - angle):
        #     result_angles.append(calculated_angles_1[-1][angle_index])
        # else:
        #     result_angles.append(calculated_angles_2[-1][angle_index])

        # total_angle_calc.append(total_angle_calc[-1] + result_angles[-1])

    blue_patch = mpatches.Patch(color='blue', label='Действительный угол')
    red_patch = mpatches.Patch(color='red', label='Рассчитанный угол gamma')
    green_patch = mpatches.Patch(color='green', label='Рассчитанный угол theta')
    purple_patch = mpatches.Patch(color='purple', label='Рассчитанный угол Psi')

    plt.plot([angles[angle_index] for angles in real_angles], 'b')
    plt.plot([angles[0] for angles in result_angles], 'r')
    plt.plot([angles[1] for angles in result_angles], 'g')
    plt.plot([angles[2] for angles in result_angles], 'purple')
    plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()

    plt.plot([real_angle[0] - angles[0] for real_angle, angles in zip(real_angles, result_angles)], 'r')
    plt.plot([real_angle[1] - angles[1] for real_angle, angles in zip(real_angles, result_angles)], 'g')
    plt.plot([real_angle[2] - angles[2] for real_angle, angles in zip(real_angles, result_angles)], 'purple')

    red_diff_patch = mpatches.Patch(color='red', label='Отклонение от действительного угла gamma')
    green_diff_patch = mpatches.Patch(color='green', label='Отклонение от действительного угла theta')
    purple_patch = mpatches.Patch(color='purple', label='Отклонение от действительного угла Psi')
    plt.legend(handles=[red_diff_patch, green_diff_patch, purple_patch], loc='upper right')
    plt.show()


    # plt.plot(total_angle_real, 'b')
    # plt.plot(total_angle_calc, 'g')
    # blue_patch = mpatches.Patch(color='blue', label='Действительный угол')
    # green_label = mpatches.Patch(color='green', label='Рассчитанный угол')
    # plt.legend(handles=[blue_patch, green_label], loc='upper right')
    # plt.show()

    # plt.plot([x - y for x, y in zip(total_angle_real, total_angle_calc)], 'g')
    # plt.show()

    # print(total_angle_real[-1] - total_angle_calc[-1])

    # plt.plot(real_angles, 'b')
    # # plt.plot([angle[0] for angle in calculated_angles_2], 'r')
    # # plt.plot([angle[1] for angle in calculated_angles_2], 'g')
    #
    # plt.show()


if __name__ == '__main__':
    main()
