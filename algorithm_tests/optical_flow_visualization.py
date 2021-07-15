import cv2
import numpy as np


def visualize_keypoints(image: np.ndarray, keypoints: list, boundaries: list, marker_type) -> np.ndarray:
    """
    Draw keypoints on previously created image
    :param image: image to draw keypoints on
    :param keypoints: list with keypoints
    :param boundaries: real world boundaries
    :param marker_type: marker type: cv2.MARKER_CROSS/cv2.MARKER_DIAMOND/...
    :return:
    """
    for p in keypoints:
        tmp = np.ndarray([2, 1])
        tmp[0] = int(p[0] - boundaries[0][0])
        tmp[1] = int(p[1] - boundaries[0][1])
        cv2.drawMarker(image, (int(tmp[0]), int(image.shape[0] - tmp[1])), (0,), marker_type)

    return image


def visualize_keypoints_without_transformation(image: np.ndarray, keypoints: list, marker_type) -> np.ndarray:
    for kp in keypoints:
        cv2.drawMarker(image, (int(kp[0]), int(kp[1])), (0,), marker_type)

    return image


def visualize_keypoints_deltas(image: np.ndarray, points_set_1: list, points_set_2: list) -> np.ndarray:
    """
    Draw lines between two keypoints sets
    :param image: Image to draw on
    :param points_set_1: First keypoint set
    :param points_set_2: Second keypoint set
    :param boundaries: Boundaries of real world coordinates
    :return:
    """
    for pt1, pt2 in zip(points_set_1, points_set_2):
        cv2.line(
            image,
            (int(pt1[0][0]), int(pt1[1][0])),
            (int(pt2[0][0]), int(pt2[1][0])),
            (255,),
            thickness=2
        )

    return image


def visualize_not_moving_coordinates(boundaries: list) -> np.ndarray:
    """
    Create image and draw not moving coordinates
    :param boundaries: Boundaries of real world coordinates
    :return:
    """
    image_center_coordinates = [
        int((boundaries[1][0] - boundaries[0][0]) / 2),
        int((boundaries[1][1] - boundaries[0][1]) / 2)
    ]

    image_size = [
        int(boundaries[1][0] - boundaries[0][0]),
        int(boundaries[1][1] - boundaries[0][1])
    ]

    image = np.ones((
        image_size[0], image_size[1]
    ), dtype=np.uint8) * 128

    cv2.line(image, (image_center_coordinates[0], 0), (image_center_coordinates[0], image_size[1] - 1), (255,))

    cv2.line(image, (0, image_center_coordinates[1]), (image_size[0] - 1, image_center_coordinates[1]), (255,))

    return image


def visualize_camera_fov(image: np.ndarray, camera_fov: list, boundaries: list, color: tuple) -> np.ndarray:
    """
    Visualize camera's field of view
    :param image: image where to draw fov
    :param camera_fov:
    :param boundaries:
    :return:
    """
    points_for_polygon = []
    for p in camera_fov:
        points_for_polygon.append([
            p[0] - boundaries[0][0],
            image.shape[0] - (p[1] - boundaries[0][1])
        ])

    points_for_polygon = np.array(points_for_polygon, np.int32)

    cv2.polylines(image, [points_for_polygon], True, color)

    return image


def visualize_optical_flow(image: np.ndarray, points_set_1: np.ndarray, points_set_2: np.ndarray,
                           calculated_points_2: np.ndarray = None) -> np.ndarray:
    """
    Visualize optical flow on provided picture
    Points must be matched and have the same amount of points
    :param image: image where to visualize optical flow
    :param points_set_1: set of points on the first image
    :param points_set_2: set of points on the second image
    :param calculated_points_2: optional parameter to visualize calculated position of the second points set
    :return:
    """

    for pt1, pt2 in zip(points_set_1, points_set_2):
        cv2.line(image, (pt1[0][0], image.shape[0] - pt1[1][0]), (pt2[0][0], pt2[1][0]), (255,), thickness=2)
        cv2.drawMarker(image, (pt1[0][0], image.shape[0] - pt1[1][0]), (0,), markerType=cv2.MARKER_CROSS)
        cv2.drawMarker(image, (pt2[0][0], image.shape[0] - pt2[1][0]), (0,), markerType=cv2.MARKER_DIAMOND)

    if calculated_points_2 is not None:
        for pt1, pt2 in zip(points_set_1, calculated_points_2):
            cv2.line(image, (pt1[0][0], image.shape[0] - pt1[1][0]), (pt2[0][0], image.shape[0] - pt2[1][0]), (0,), thickness=2)
            cv2.drawMarker(image, (pt1[0][0], image.shape[0] - pt1[1][0]), (0,), markerType=cv2.MARKER_CROSS)
            cv2.drawMarker(image, (pt2[0][0], image.shape[0] - pt2[1][0]), (0,), markerType=cv2.MARKER_DIAMOND)

    return image
