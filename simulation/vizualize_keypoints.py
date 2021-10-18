import cv2
import numpy as np


def get_color(px_position: list, canvas_size: tuple) -> tuple:
    x = px_position[0]
    y = px_position[1]
    img_shape_x = canvas_size[1]
    img_shape_y = canvas_size[0]
    bottom_right_corner = (60, 100, 255)
    pix_value = np.uint8([[[int(y/img_shape_y*bottom_right_corner[0])*2, int((img_shape_x - x)/img_shape_x*bottom_right_corner[1] + 100), int((img_shape_y - y)/img_shape_y*(bottom_right_corner[2]))]]])
    color = cv2.cvtColor(pix_value, cv2.COLOR_HSV2BGR)[0][0]
    return int(color[0]), int(color[1]), int(color[2])


def visualize_keypoints(image: np.ndarray, keypoints: list, marker_type) -> np.ndarray:
    """
    Draw keypoints on previously created image
    :param image: image to draw keypoints on
    :param keypoints: list with keypoints
    :param boundaries: real world boundaries
    :param marker_type: marker type: cv2.MARKER_CROSS/cv2.MARKER_DIAMOND/...
    :return:
    """
    img = image.copy()
    for p in keypoints:
        cv2.drawMarker(img, (int(p[0]), int(p[1])), get_color(p, image.shape), marker_type, thickness=4)

    return img


def draw_optical_flow_img(previous_mask: list, current_mask: list, previous_kp: list, current_kp: list, canvas: np.ndarray):
    img = canvas.copy()
    for (visible_1, visible_2, kp_1, kp_2) in zip(previous_mask, current_mask, previous_kp, current_kp):
        if visible_1 and visible_2:
            cv2.line(
                img,
                (int(kp_1[0]), int(kp_1[1])),
                (int(kp_2[0]), int(kp_2[1])),
                (255, 255, 255),
                thickness=4
            )
    return img