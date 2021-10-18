import numpy as np

from simulation.camera_v2 import Camera


def calculate_keypoints_on_image(keypoints: list, camera: Camera):
    visible_keypoints = []
    all_keypoints = []
    mask = []
    R = camera.R
    for keypoint in keypoints:
        position_delta = keypoint - camera.position
        nominator_x = R[0, 0] * position_delta[0] + R[1, 0] * position_delta[1] + R[2, 0] * position_delta[2]
        denominator_x = R[0, 2] * position_delta[0] + R[1, 2] * position_delta[1] + R[2, 2] * position_delta[2]

        nominator_y = R[0, 1]*position_delta[0] + R[1, 1]*position_delta[1] + R[2, 1]*position_delta[2]
        denominator_y = R[0, 2]*position_delta[0] + R[1, 2]*position_delta[1] + R[2, 2]*position_delta[2]

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

    return (mask, visible_keypoints, all_keypoints)
