import math

import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.camera_v2 import Camera

from simulation.calculate_keypoint_image_position import calculate_keypoints_on_image
from simulation.camera_movement import LinearMovement, SinMovement
from simulation.generate_image import generate_canvas
from simulation.kp_generator import generate_keypoints_equal
from simulation.rotation_matrix import calculate_rotation_matrix
from simulation.transform_keypoints import convert_keypoints_to_px
from simulation.vizualize_keypoints import visualize_keypoints, draw_optical_flow


def get_keypoints_and_img(canvas, keypoints, camera, coeffs):
    mask, visible_keypoints, keypoints_px = calculate_keypoints_on_image(keypoints, camera)
    px_camera_keypoints = convert_keypoints_to_px(visible_keypoints, coeffs, canvas.shape)
    img = visualize_keypoints(canvas, px_camera_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (800, 800))

    return mask, visible_keypoints, keypoints_px, result_image


def draw_keypoints_on_img(canvas, keypoints, coeffs):
    px_camera_keypoints = convert_keypoints_to_px(keypoints, coeffs, canvas.shape)
    img = visualize_keypoints(canvas, px_camera_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (800, 800))

    return result_image


def show_image(win_name, img):
    cv2.imshow(win_name, img)
    cv2.waitKey(50)


def get_keypoints_both_pictures(all_kp: list, kp1: list, kp2: list, mask1: list, mask2: list) -> (list, list, list):
    visible_kp_1 = []
    visible_kp_2 = []
    real_visible_kp = []
    for (is_visible_1, is_visible_2, kp_1, kp_2, real_kp) in zip(mask1, mask2, kp1, kp2, all_kp):
        if is_visible_1 and is_visible_2:
            visible_kp_1.append(kp_1)
            visible_kp_2.append(kp_2)
            real_visible_kp.append(real_kp)

    visible_kp_1 = np.int32(visible_kp_1)
    visible_kp_2 = np.int32(visible_kp_2)
    return visible_kp_1, visible_kp_2, real_visible_kp


def get_abs_diff_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    r = abs(R1 - R2)
    return sum(sum(r))


def calculate_angles(R: np.ndarray) -> list:
    rad_to_deg = lambda x: x * 180 / math.pi
    return [
        rad_to_deg(math.atan(R[2, 1] / R[2, 2])),
        rad_to_deg(-math.asin(R[2, 0])),
        rad_to_deg(math.atan(R[1, 0]/R[0, 0]))
    ]


def main():
    initial_position = np.zeros([3, 1])
    initial_position[0] = 500
    initial_position[1] = 500
    initial_position[2] = 100

    final_position = np.array([600, 600, 100])

    initial_rotation = [0, 0, 0]
    initial_rotation_matrix = calculate_rotation_matrix(
        initial_rotation[0] * math.pi / 180,
        initial_rotation[1] * math.pi / 180,
        initial_rotation[2] * math.pi / 180
    )

    final_rotation = [0, 0, 0]

    resolution = [600, 600]
    fov = [60, 60]
    f = 40 / 1000

    camera = Camera(initial_position, initial_rotation_matrix, f, fov, resolution)
    canvas = generate_canvas(1500, 1500)
    camera_canvas = generate_canvas(camera.resolution[0], camera.resolution[1])

    keypoints = generate_keypoints_equal(np.array([1500, 1500]), x_keypoint_amount=80, y_keypoint_amount=80)

    coeffs = [1, 1]
    px_keypoints = convert_keypoints_to_px(keypoints, coeffs, canvas.shape)

    img = visualize_keypoints(canvas, px_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (800, 800))
    cv2.imshow('test', result_image)
    cv2.waitKey(1)

    movement = LinearMovement(initial_position, final_position, initial_rotation, final_rotation, 10)
    amplitude_x = np.ndarray([3, 1])
    amplitude_x[0] = 0
    amplitude_x[1] = 0
    amplitude_x[2] = 0
    amplitude_w = np.array([10 * math.pi / 180, 10 * math.pi / 180, 0])

    movement = SinMovement(amplitude_x, amplitude_w, math.pi*6, 50)
    mask, visible_keypoints, keypoints_px, result_image = get_keypoints_and_img(canvas, keypoints, camera, coeffs)
    show_image('camera', result_image)

    current_kp = keypoints_px

    current_mask = mask

    r = np.eye(3)

    angles = []
    real_angles = []

    for _ in movement.move_camera(camera):
        mask, visible_keypoints, keypoints_px, result_image = get_keypoints_and_img(canvas, keypoints, camera, coeffs)
        show_image('camera', result_image)

        previous_kp = current_kp.copy()
        current_kp = keypoints_px.copy()

        previous_mask = current_mask.copy()
        current_mask = mask.copy()

        prev_img_kp, current_img_kp, real_kp = get_keypoints_both_pictures(keypoints, current_kp, previous_kp,
                                                                           current_mask, previous_mask)
        img = draw_keypoints_on_img(canvas, real_kp, coeffs)
        show_image('both_pictures', img)

        E, mask = cv2.findEssentialMat(prev_img_kp, current_img_kp, camera.internal_matrix, method=cv2.RANSAC)
        R1, R2, t = cv2.decomposeEssentialMat(E)
        tmp_rotation_matrix_1 = np.dot(R1, r)
        tmp_rotation_matrix_2 = np.dot(R2, r)
        print(f'angles1: {calculate_angles(tmp_rotation_matrix_1)}, angles2: {calculate_angles(tmp_rotation_matrix_2)}')
        delta1 = get_abs_diff_mat(tmp_rotation_matrix_1, r)
        delta2 = get_abs_diff_mat(tmp_rotation_matrix_2, r)

        print(f'delta1: {1}, delta2: {2}', delta1, delta2)

        if delta1 < delta2:
            r = tmp_rotation_matrix_1.copy()
        else:
            r = tmp_rotation_matrix_2.copy()

        real_angles.append(calculate_angles(camera.ufo.rotation_matrix))
        angles.append(calculate_angles(r))

    blue_patch = mpatches.Patch(color='blue', label='Действительный угол')
    red_patch = mpatches.Patch(color='red', label='Рассчитанный угол gamma')
    green_patch = mpatches.Patch(color='green', label='Рассчитанный угол theta')
    purple_patch = mpatches.Patch(color='purple', label='Рассчитанный угол Psi')

    plt.plot([angle[0] for angle in real_angles], 'b')
    plt.plot([-angle[0] for angle in angles], 'r')
    plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()

    plt.plot([angle[1] for angle in real_angles], 'b')
    plt.plot([-angle[1] for angle in angles], 'r')
    plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()

    plt.plot([angle[2] for angle in real_angles], 'b')
    plt.plot([angle[2] for angle in angles], 'r')
    plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()

    plt.plot([real_angle[0] + angle[0] for (real_angle, angle) in zip(real_angles, angles)], 'b')
    plt.plot([real_angle[1] + angle[1] for (real_angle, angle) in zip(real_angles, angles)], 'r')
    plt.plot([real_angle[2] - angle[2] for (real_angle, angle) in zip(real_angles, angles)], 'g')
    plt.show()


if __name__ == '__main__':
    main()
