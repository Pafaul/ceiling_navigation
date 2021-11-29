import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv2.VideoCapture('./videos/calibration.mp4')  # 'http://172.16.0.212:8080/video')
captured = 0
required = 100
mtx_collection = []
dist_collection = []

images = []

while True:
    res, img = cap.read()

    img = cv2.resize(img, (1280, 720))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('cap', gray)
    cv2.waitKey(1)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (4, 4), None)

    # If found, add object points, image points (after refining them)
    if ret:
        images.append(gray)

        captured += 1
        print(f'captured: {captured}')
        print(f'Left: {required - captured}')
        if captured == required:
            break
    else:
        print('cannot find')

for gray in images:
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    objpoints = []
    imgpoints = []
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    cv2.imshow('chess', cv2.resize(img, (1600, 900)))
    cv2.waitKey(1)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    mtx_collection.append(mtx)
    dist_collection.append(dist)

print(sum(mtx_collection) / required)
print(sum(dist_collection) / required)

cv2.destroyAllWindows()
