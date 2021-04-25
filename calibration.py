import numpy as np
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

a, b = 6, 8
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((a*b,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = ['template_1.jpg']

camera = PiCamera(resolution='HD', framerate=30, sensor_mode=6)
# cap = cv2.VideoCapture(0) #'http://172.16.0.212:8080/video')
time.sleep(10)
captured = 0
required = 20
mtx_collection = []
dist_collection = []

images = []

rawCapture = PiRGBArray(camera, size=(1280,720))
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    gray = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
    rawCapture.truncate(0)
    cv2.imshow('cap', gray)
    cv2.waitKey(1)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (a,b),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        images.append(gray)
       
        captured += 1
        print(f'Left: {required - captured}')
        if (captured == required): 
            break
    else:
        print('Cannot find chessboard.')
        print(corners)

for gray in images:
    ret, corners = cv2.findChessboardCorners(gray, (a,b),None)
    objpoints = []
    imgpoints = []
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(gray, (a,b), corners2,ret)
    cv2.imshow('chess',img)
    cv2.waitKey(100)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    mtx_collection.append(mtx)
    dist_collection.append(dist)

print(sum(mtx_collection)/required)
print(sum(dist_collection)/required)

cv2.destroyAllWindows()
