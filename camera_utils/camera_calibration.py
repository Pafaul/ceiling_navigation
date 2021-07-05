import cv2
import numpy as np


def calibrate_camera(calibration_image: str = None, calibration_file: str = None, separator: str = ' '):
    """
    Calibrate camera if no calibration file specified
    :param calibration_image: Path to image to calibrate camera with
    :param calibration_file: File with calibration parameters
    :param separator: Separator used in calibration_file
    :return:
    """
    if calibration_image is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        img = cv2.imread(calibration_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img, (7, 6), None)

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
        return mtx, dist

    if calibration_file is not None:
        with open(calibration_file, 'r') as f:
            lines = f.readlines()
            dist = np.array([float(n) for n in lines[-1].strip().split(separator)])
            tmp_arr = []
            for line in lines[:-1]:
                tmp_arr.append([float(n) for n in line.strip().split(separator)])
            mtx = np.array(tmp_arr)

            return mtx, dist
