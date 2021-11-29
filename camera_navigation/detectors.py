from cv2 import cv2


class BasicDetector:
    def get_keypoints(self, img):
        raise NotImplementedError()


class ORBDetector(BasicDetector):
    def __init__(self):
        self.detector = cv2.ORB_create(
            scaleFactor=2,
            nfeatures=150,
            edgeThreshold=50,
            fastThreshold=20
        )

    def get_keypoints(self, img):
        keypoints = self.detector.detect(img, None)
        keypoints, des = self.detector.compute(img, keypoints)
        return keypoints, des


class SIFTDetector(BasicDetector):
    def __init__(self):
        self.detector = cv2.SIFT_create(
            contrastThreshold=0.1
        )

    def get_keypoints(self, img):
        keypoints, des = self.detector.detectAndCompute(img, None)
        return keypoints, des
