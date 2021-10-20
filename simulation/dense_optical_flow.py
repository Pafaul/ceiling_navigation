import math

import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from simulation.camera_movement import LinearMovement, SinMovement
from simulation.camera_v2 import Camera
from simulation.generate_image import generate_canvas
from simulation.rotation_matrix import calculate_rotation_matrix
from simulation.vizualize_keypoints import visualize_keypoints, draw_optical_flow_img


def calculate_angles(R: np.ndarray) -> list:
    rad_to_deg = lambda x: x * 180 / math.pi
    return [
        rad_to_deg(math.atan(R[2, 1] / R[2, 2])),
        rad_to_deg(-math.asin(R[2, 0])),
        rad_to_deg(math.atan(R[1, 0]/R[0, 0]))
    ]


def get_abs_diff_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    r = abs(R1 - R2)
    return sum(sum(r))


def generate_keypoints_on_image(camera: Camera, x_keypoints = 6, y_keypoints = 4) -> list:
    keypoints = []
    delta_x = camera.resolution[0] / (x_keypoints + 1)
    delta_y = camera.resolution[1] / (y_keypoints + 1)
    for i in range(x_keypoints):
        for j in range(y_keypoints):
            tmp = np.ndarray([2, 1])
            tmp[0] = int(delta_x + i * delta_x)
            tmp[1] = int(delta_y + j * delta_y)
            keypoints.append(tmp)
    return keypoints


def px_to_obj(camera: Camera, keypoints: list) -> list:
    r = camera.R.copy()
    u = camera.resolution[0] / 2 * camera.px_mm[0]
    v = camera.resolution[1] / 2 * camera.px_mm[1]
    kp_obj = []
    for kp in keypoints:
        obj_position = np.ndarray([3, 1])
        obj_position[2] = 0

        obj_position[0] = camera.position[0] + (obj_position[2] - camera.position[2]) * \
            (r[0, 0] * (kp[0] * camera.px_mm[0] - u) + r[0, 1] * (kp[1] * camera.px_mm[1] - v) - r[0, 2] * camera.f) / \
            (r[2, 0] * (kp[0] * camera.px_mm[0] - u) + r[2, 1] * (kp[1] * camera.px_mm[1] - v) - r[2, 2] * camera.f)

        obj_position[1] = camera.position[1] + (obj_position[2] - camera.position[2]) * \
            (r[1, 0] * (kp[0] * camera.px_mm[0] - u) + r[1, 1] * (kp[1] * camera.px_mm[1] - v) - r[1, 2] * camera.f) / \
            (r[2, 0] * (kp[0] * camera.px_mm[0] - u) + r[2, 1] * (kp[1] * camera.px_mm[1] - v) - r[2, 2] * camera.f)
        kp_obj.append(obj_position)

    return kp_obj


def obj_to_px(camera: Camera, keypoints: list) -> list:
    visible_keypoints = []
    all_keypoints = []
    mask = []
    R = camera.R.copy()
    for keypoint in keypoints:
        position_delta = keypoint - camera.position
        nominator_x = R[0, 0] * position_delta[0] + R[1, 0] * position_delta[1] + R[2, 0] * position_delta[2]
        denominator_x = R[0, 2] * position_delta[0] + R[1, 2] * position_delta[1] + R[2, 2] * position_delta[2]

        nominator_y = R[0, 1] * position_delta[0] + R[1, 1] * position_delta[1] + R[2, 1] * position_delta[2]
        denominator_y = R[0, 2] * position_delta[0] + R[1, 2] * position_delta[1] + R[2, 2] * position_delta[2]

        x = camera.resolution[0] / 2 - camera.f * (nominator_x / denominator_x) / camera.px_mm[0]
        y = camera.resolution[1] / 2 - camera.f * (nominator_y / denominator_y) / camera.px_mm[1]
        all_keypoints.append(np.zeros([2, 1]))
        all_keypoints[-1][0] = x
        all_keypoints[-1][1] = y
        if (0 <= x <= camera.resolution[0]) and (0 <= y <= camera.resolution[1]):
            mask.append(True)
            visible_keypoints.append(keypoint)
        else:
            mask.append(False)

    return mask, visible_keypoints, all_keypoints


def visualize_optical_flow(canvas: np.ndarray, previous_keypoints: list, current_keypoints: list,
                           previous_mask: list, current_mask: list) -> np.ndarray:
    img = canvas.copy()
    img = visualize_keypoints(img, previous_keypoints, cv2.MARKER_CROSS)
    img = visualize_keypoints(img, current_keypoints, cv2.MARKER_DIAMOND)
    img = draw_optical_flow_img(previous_mask, current_mask, previous_keypoints, current_keypoints, img)
    return img


def main():
    moving_points = 100

    initial_position = np.zeros([3, 1])
    initial_position[0] = 0
    initial_position[1] = 0
    initial_position[2] = 100

    final_position = np.array([0, 0, 100])

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

    keypoints = generate_keypoints_on_image(camera, x_keypoints=6, y_keypoints=4)

    # movement = LinearMovement(initial_position, final_position, initial_rotation, final_rotation, moving_points)
    amplitude_x = np.ndarray([3, 1])
    amplitude_x[0] = 0
    amplitude_x[1] = 0
    amplitude_x[2] = 0
    amplitude_w = np.array([0 * math.pi / 180, 0 * math.pi / 180, 10 * math.pi / 180])

    movement = SinMovement(amplitude_x, amplitude_w, math.pi*10, moving_points)

    mask = [True] * len(keypoints)

    r = np.eye(3)

    angles = []
    real_angles = []

    move_camera = movement.move_camera(camera)
    for i in range(moving_points-1):

        canvas = generate_canvas(600, 600)

        img = visualize_keypoints(canvas, keypoints, cv2.MARKER_DIAMOND)
        result_image = cv2.resize(img, (800, 800))
        cv2.imshow('first', result_image)
        cv2.waitKey(1)

        obj_keypoints = px_to_obj(camera, keypoints)

        next(move_camera)

        currnet_mask, visible_keypoints, all_keypoints = obj_to_px(camera, obj_keypoints)
        px_keypoints = []
        for (visible, kp) in zip(currnet_mask, all_keypoints):
            if visible:
                px_keypoints.append(kp)

        img = visualize_keypoints(canvas, px_keypoints, cv2.MARKER_CROSS)

        result_image = cv2.resize(img, (800, 800))
        cv2.imshow('second', result_image)
        cv2.waitKey(1)

        img = visualize_optical_flow(canvas, keypoints, all_keypoints, mask, currnet_mask)
        cv2.imshow('visual flow', img)
        cv2.waitKey(10)

        E, mask = cv2.findEssentialMat(np.int32(keypoints), np.int32(px_keypoints), camera.internal_matrix, method=cv2.RANSAC)
        R1, R2, t = cv2.decomposeEssentialMat(E)
        tmp_rotation_matrix_1 = np.dot(R1, r)
        tmp_rotation_matrix_2 = np.dot(R2, r)
        print(f'angles1: {calculate_angles(tmp_rotation_matrix_1)}, angles2: {calculate_angles(tmp_rotation_matrix_2)}')
        delta1 = get_abs_diff_mat(tmp_rotation_matrix_1, r)
        delta2 = get_abs_diff_mat(tmp_rotation_matrix_2, r)

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
    plt.plot([angle[0] for angle in angles], 'r')
    plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()

    plt.plot([angle[1] for angle in real_angles], 'b')
    plt.plot([angle[1] for angle in angles], 'r')
    plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()

    plt.plot([angle[2] for angle in real_angles], 'b')
    plt.plot([angle[2] for angle in angles], 'r')
    plt.legend(handles=[blue_patch, red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()

    plt.plot([real_angle[0] - angle[0] for (real_angle, angle) in zip(real_angles, angles)], 'b')
    plt.plot([real_angle[1] - angle[1] for (real_angle, angle) in zip(real_angles, angles)], 'r')
    plt.plot([real_angle[2] - angle[2] for (real_angle, angle) in zip(real_angles, angles)], 'g')
    plt.show()


main()