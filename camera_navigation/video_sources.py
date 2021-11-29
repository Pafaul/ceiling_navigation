from typing import Tuple

import numpy as np
from cv2 import cv2


class BasicVideoSource:
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError()


class VideoCaptureSource(BasicVideoSource):
    def __init__(self, config: dict):
        self.source = cv2.VideoCapture(config['video_source'])

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        while self.source.isOpened():
            success, frame = self.source.read()
            yield success, frame
