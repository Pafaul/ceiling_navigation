import cv2
import numpy as np

from simulation.transform_keypoints import convert_keypoints_to_px


def get_color(px_position: list, canvas_size: tuple) -> tuple:
    x = px_position[0]
    y = px_position[1]
    img_shape_x = canvas_size[1]
    img_shape_y = canvas_size[0]
    bottom_right_corner = (60, 100, 255)
    pix_value = np.uint8([[[int(y/img_shape_y*bottom_right_corner[0])*2, int((img_shape_x - x)/img_shape_x*bottom_right_corner[1] + 100), int((img_shape_y - y)/img_shape_y*(bottom_right_corner[2]))]]])
    color = cv2.cvtColor(pix_value, cv2.COLOR_HSV2BGR)[0][0]
    return int(color[0]), int(color[1]), int(color[2])


def draw_keypoints_on_img(canvas, keypoints, visualization_config):
    px_camera_keypoints = convert_keypoints_to_px(
        keypoints,
        visualization_config['coefficients'],
        canvas.shape
    )
    return visualize_keypoints(canvas, px_camera_keypoints, cv2.MARKER_DIAMOND)


def draw_optical_flow(canvas: np.ndarray, mask1: list, mask2: list, kp1: list, kp2: list, visualization_config: dict):
    keypoints_to_visualize_1 = []
    keypoints_to_visualize_2 = []

    for (visible_1, visible_2, kp_1, kp_2) in zip(mask1, mask2, kp1, kp2):
        if visible_1 and visible_2:
            keypoints_to_visualize_1.append(kp_1)
            keypoints_to_visualize_2.append(kp_2)

    img = canvas.copy()
    img = draw_optical_flow_img(mask1, mask2, kp1, kp2, img)
    img = visualize_keypoints(img, keypoints_to_visualize_1, cv2.MARKER_CROSS)
    img = visualize_keypoints(img, keypoints_to_visualize_2, cv2.MARKER_DIAMOND)
    return img


def show_image(win_name, img, visualization_config: dict):
    if visualization_config['visualization_enabled']:
        img = cv2.resize(img, (
            visualization_config['visualization_resolution'][0],
            visualization_config['visualization_resolution'][1]
        ))
        cv2.imshow(win_name, img)
        cv2.waitKey(
            visualization_config['visualization_pause']
        )


def visualize_keypoints(image: np.ndarray, keypoints: list, marker_type) -> np.ndarray:
    """
    Draw keypoints on previously created image
    :param image: image to draw keypoints on
    :param keypoints: list with keypoints
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
                thickness=2
            )
    return img
