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
    mask, visible_keypoints = calculate_keypoints_on_image(keypoints, camera)
    px_camera_keypoints = convert_keypoints_to_px(visible_keypoints, coeffs, canvas.shape)
    img = visualize_keypoints(canvas, px_camera_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (1000, 1000))

    return (mask, visible_keypoints, result_image)

def main():
    initial_position = np.zeros([3, 1])
    initial_position[0] = 0
    initial_position[1] = 0
    initial_position[2] = 100

    final_position = np.array([1500, 1500, 100])

    initial_rotation = [0, 0, 0]
    final_rotation = [0, 0, 30]

    resolution = [600, 600]
    fov = [30, 30]
    f = 40 / 1000

    camera = Camera(initial_position, f, fov, resolution)
    canvas = generate_canvas(1500, 1500)

    keypoints = generate_keypoints_equal(np.array([1500, 1500]), x_keypoint_amount=40, y_keypoint_amount=40)

    coeffs = [1, 1]
    px_keypoints = convert_keypoints_to_px(keypoints, coeffs, canvas.shape)

    img = visualize_keypoints(canvas, px_keypoints, cv2.MARKER_DIAMOND)
    result_image = cv2.resize(img, (1000, 1000))
    cv2.imshow('test', result_image)
    cv2.waitKey(1)

    movement = LinearMovement(initial_position, final_position, initial_rotation, final_rotation, 10)

    for _ in movement.move_camera(camera):
        mask, visible_keypoints, result_image = get_keypoints_and_img(canvas, keypoints, camera, coeffs)
        cv2.imshow('camera', result_image)
        cv2.waitKey(1000)


main()
