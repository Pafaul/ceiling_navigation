import math

import cv2
import numpy as np
from algorithm_tests.camera_v2 import Camera

from simulation.calculate_keypoint_image_position import calculate_keypoints_on_image
from simulation.camera_movement import LinearMovement
from simulation.generate_image import generate_canvas
from simulation.kp_generator import generate_keypoints_equal
from simulation.rotation_matrix import calculate_rotation_matrix
from simulation.transform_keypoints import convert_keypoints_to_px
from simulation.vizualize_keypoints import visualize_keypoints


def get_keypoints_and_img(canvas, keypoints, camera, coeffs):
    mask, visible_keypoints, keypoints_px = calculate_keypoints_on_image(keypoints, camera)
    px_camera_keypoints = convert_keypoints_to_px(visible_keypoints, coeffs, canvas.shape)
    img = visualize_keypoints(canvas, px_camera_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (1000, 1000))

    return mask, visible_keypoints, keypoints_px, result_image


def draw_keypoints_on_img(canvas, keypoints, coeffs):
    px_camera_keypoints = convert_keypoints_to_px(keypoints, coeffs, canvas.shape)
    img = visualize_keypoints(canvas, px_camera_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (1000, 1000))

    return result_image


def show_image(win_name, img):
    cv2.imshow(win_name, img)
    cv2.waitKey(1000)


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


def main():
    initial_position = np.zeros([3, 1])
    initial_position[0] = 0
    initial_position[1] = 0
    initial_position[2] = 100

    final_position = np.array([1000, 1000, 100])

    initial_rotation = [0, 0, 0]
    final_rotation = [0, 0, 90]

    resolution = [600, 600]
    fov = [30, 30]
    f = 40 / 1000

    camera = Camera(initial_position, f, fov, resolution)
    camera.R = calculate_rotation_matrix(initial_rotation[0], initial_rotation[1], initial_rotation[2])
    canvas = generate_canvas(1500, 1500)

    keypoints = generate_keypoints_equal(np.array([1500, 1500]), x_keypoint_amount=40, y_keypoint_amount=40)

    coeffs = [1, 1]
    px_keypoints = convert_keypoints_to_px(keypoints, coeffs, canvas.shape)

    img = visualize_keypoints(canvas, px_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (1000, 1000))
    cv2.imshow('test', result_image)
    cv2.waitKey(1)

    movement = LinearMovement(initial_position, final_position, initial_rotation, final_rotation, 10)
    mask, visible_keypoints, keypoints_px, result_image = get_keypoints_and_img(canvas, keypoints, camera, coeffs)
    show_image('camera', result_image)

    current_kp = keypoints_px
    previous_kp = []

    current_mask = mask
    previous_mask = []

    r = camera.R.copy()

    for _ in movement.move_camera(camera):
        mask, visible_keypoints, keypoints_px, result_image = get_keypoints_and_img(canvas, keypoints, camera, coeffs)
        show_image('camera', result_image)

        previous_kp = current_kp
        current_kp = keypoints_px

        previous_mask = current_mask
        current_mask = mask

        prev_img_kp, current_img_kp, real_kp = get_keypoints_both_pictures(keypoints, current_kp, previous_kp,
                                                                           current_mask, previous_mask)
        img = draw_keypoints_on_img(canvas, real_kp, coeffs)
        show_image('both_pictures', img)

        E, mask = cv2.findEssentialMat(current_img_kp, prev_img_kp, camera.internal_matrix, method=cv2.RANSAC)
        R1, R2, t = cv2.decomposeEssentialMat(E)
        tmp_rotation_matrix_1 = np.dot(R1, r)
        tmp_rotation_matrix_2 = np.dot(R2, r)
        delta1 = get_abs_diff_mat(tmp_rotation_matrix_1, r)
        delta2 = get_abs_diff_mat(tmp_rotation_matrix_2, r)

        if delta1 < delta2:
            r = tmp_rotation_matrix_1
        else:
            r = tmp_rotation_matrix_2

        print(r)
        print(camera.R)


if __name__ == '__main__':
    main()
