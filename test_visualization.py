import cv2
import numpy as np

def create_empty_canvas(height: int, width: int):
    return np.zeros((height, width, 3), np.uint8)

def colorize_canvas(img: np.ndarray):
    top_left_corner = (0, 255, 255)
    bottom_right_corner = (50, 100, 155)
    keypoints = []
    for i in range(10):
        x, y = int(img.shape[0] * (i + 1) / 10), int(img.shape[1] * (i + 1) / 10)
        pix_value = np.uint8([[[int(y/img.shape[1]*bottom_right_corner[0])*2, int((img.shape[0] - x)/img.shape[0]*bottom_right_corner[1] + 100), int((img.shape[1] - y)/img.shape[1]*(bottom_right_corner[2]) + 100)]]])
        color = cv2.cvtColor(pix_value, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_DIAMOND, thickness=4)

    return img

img = create_empty_canvas(1000, 1000)
img = colorize_canvas(img)
cv2.imshow('test', img)
cv2.waitKey(0)