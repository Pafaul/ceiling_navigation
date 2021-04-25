
from position_calculation.get_coordinates_from_global_map import get_coordinates_from_global_map
import time
from utils.init_functions import get_config, get_thresholds, initial_stage, initialize_algorithms, initialize_kalman_filter
import cv2
import pickle

from math import acos, pi, atan2, sqrt
from sys import argv

import numpy as np
import matplotlib.pyplot as plt

from camera_utils.match_keypoints import match_keypoints
from camera_utils.get_keypoints import get_keypoints_from_image
from position_calculation.get_position_deltas import get_position_deltas
from lsm.least_square_method import seq_lsm

from picamera.array import PiRGBArray

measurments_descriptions = [
    "X",
    "Y",
    "Z",
    "angle X",
    "angle Y",
    "angle Z",
    "dt"
]

error_descriptions = [
    "D_X",
    "D_Y",
    "D_Z",
    "D_A_X",
    "D_A_Y",
    "D_A_Z",
    "D_T"
]

FRAMES_NAVIGATION = 1
GLOBAL_MAP_NAVIGATION = 2
init_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
dim_z = len(init_x)
p_x = np.array([0.5, 0.5, 0.5, 1, 1, 1, 0.16, 0.16, 0.16, 0.01, 0.01, 0.01])
r_x = np.array([50, 50, 50, 10, 10, 10, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05])

def main():
    filename = argv[1]
    config, debug = get_config(filename)

    camera_matrix, dist, camera, camera_distance_calculator = initial_stage(config)
    descriptor, matcher = initialize_algorithms(config)
    fk = initialize_kalman_filter(init_x, dim_z, p_x, r_x)
    threshold_config = get_thresholds(config)

    tmp_img_current = None
    tmp_des_current, tmp_des_previous = None, None
    tmp_kp_current, tmp_kp_previous = None, None
    global_pts, global_des = None, None

    Z = []
    errors = []
    delta_time = []
    current_method = FRAMES_NAVIGATION
    
    if debug['dbg_mode']:
        current_time = 0
        finish_time = float(debug['time_to_run'])
        start_time = time.time()

        i = 0
        rawCapture = PiRGBArray(camera, size=(1280,720))
        for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
            if (i > finish_time):
                break
            tmp_img_current = frame.array
            tmp_img_current = cv2.cvtColor(tmp_img_current, cv2.COLOR_BGR2GRAY)
            dt = time.time() - start_time

            start_time = time.time()
            tmp_kp_current, tmp_des_current = get_keypoints_from_image(tmp_img_current, descriptor)
            tmp_kp_img = None
            tmp_kp_img = cv2.drawKeypoints(tmp_img_current, tmp_kp_current, tmp_kp_img)
            cv2.imshow('cap', tmp_kp_img)
            cv2.waitKey(1)

            if tmp_kp_previous is not None and tmp_kp_current is not None:
                if current_method == FRAMES_NAVIGATION:
                    tmp_pts_current, tmp_pts_previous = match_keypoints(tmp_kp_current, tmp_des_current, tmp_kp_previous,
                                                                    tmp_des_previous, matcher)
                    if (len(tmp_pts_current)) < threshold_config['matcher_threshold']:
                        print('Not enough matches')
                        rawCapture.truncate(0)
                        continue
                    current_measures, measures_error = get_position_deltas(tmp_pts_previous, tmp_pts_current, camera_matrix, config=threshold_config)
                    if len(current_measures) == 0:
                        rawCapture.truncate(0)
                        continue
                    current_measures[6] = dt
                    fk(deltas=current_measures)
                    Z.append(current_measures.copy())
                    errors.append(measures_error.copy())
                    i += 1
                    print(f'Image: {i}')

                else:
                    tmp_pts_current, global_pts = match_keypoints(tmp_kp_current, tmp_des_current, global_pts, global_des, matcher)
            if tmp_kp_current is not None and tmp_des_current is not None:
                tmp_kp_previous = tmp_kp_current.copy()
                tmp_des_previous = tmp_des_current.copy()
                delta_time.append(dt)
                current_time += dt
            
            

        for index in range(6,7):
            plt.plot([t for t in range(len(Z))], [y[index] for y in Z])
            plt.xlabel('Номер шага')
            plt.ylabel(measurments_descriptions[index])
            plt.show()

        # for index in range(7):
        #     plt.plot([t for t in range(len(errors))], [y[index] for y in errors])
        #     plt.xlabel('Номер шага')
        #     plt.ylabel(error_descriptions[index])
        #     plt.show()

    fk.show()


def not_main_seq():
    img1 = cv2.imread('./images/1_2.jpg', 0)
    img2 = cv2.imread('./images/1_1.jpg', 0)
    # orb = cv2.ORB_create()
    orb = cv2.SIFT_create(contrastThreshold=0.1)
    # orb = cv2.SIFT_create()
    # BFMatcher with default params
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # bf = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher()

    kp1, des1 = orb.detectAndCompute(img1, None)
    start_time = time.time()
    kp2, des2 = orb.detectAndCompute(img2, None)

    matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    camera_matrix = np.array([
        [2.20577589e+03, 0.00000000e+00, 4.01509158e+02],
        [0.00000000e+00, 2.73065460e+03, 3.01560236e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)

    X, mask, error_EV, error_DP = seq_lsm(pts1, pts2, R)

    pts1 = np.int32(pts1[mask])
    pts2 = np.int32(pts2[mask])

    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
    retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)

    X, mask, error_EV, error_DP = seq_lsm(pts1, pts2, R)

    angle_x = atan2(R[2][1], R[2][2])
    angle_y = atan2(-R[2][0], (R[2][1] ** 2 + R[2][2] ** 2) ** 0.5)
    angle_z = atan2(R[1][0], R[0][0])
    toDeg = lambda x: x * 180 / pi
    delta = time.time() - start_time
    print(f'time of execution: {delta}')
    print(f'delta in pixels: {X}')
    print(f'KP ok: {mask.count(True)}')
    print(f'angles:\nox: {toDeg(angle_x)}, oy: {toDeg(angle_y)}, oz: {toDeg(angle_z)}')
    print(error_EV)
    print(error_DP)

def pickle_kp(kps):
    tmp = []
    for kp in kps:
        tmp.append(
            (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        )

    return tmp

def depickle_kp(file_kp, file_des):
    kp1, des1 = [], []
    with open(file_kp, 'rb') as f:
        tmp = pickle.load(f)
        for point in tmp:
            kp1.append(
                cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], 
                            _response=point[3], _octave=point[4], _class_id=point[5])
            )

    with open(file_des, 'rb') as f:
        des1 = pickle.load(f)
    return kp1, des1
    

def not_main_map(kp1, des1, orb, bf):
#    img1 = cv2.imread('./images/big_image.jpg', 0)
    img2 = cv2.imread('./images/little_image_2.png', 0)

    start_time = time.time()
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    dst = get_coordinates_from_global_map(kp1, des1, kp2, des2, bf, img2.shape)

    delta = time.time() - start_time
    # img1 = cv2.polylines(img1,[np.int32(dst)],True,255,10, cv2.LINE_AA)
    
    toDeg = lambda x: x*180/pi
    print(f'time of execution: {delta}')
    # cv2.imwrite('res.png', img1)
    x, y = 0, 0
    for p in dst: 
        x += p[0][0]
        y += p[0][1]
    x /= len(dst)
    y /= len(dst)
    print(x, y)
    h,w = img2.shape
    pt1 = dst[0][0]
    pt2 = dst[1][0]
    x = pt2[0]-pt1[0]
    y = pt2[1]-pt1[1]
    alpha = acos(h*y/(h*sqrt(x**2+y**2)))
    print(alpha)
    return delta


if __name__ == '__main__':
    kp1, des1 = depickle_kp('big_img_kp', 'big_img_des')

    orb = cv2.SIFT_create(contrastThreshold=0.1)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    dt = []
    for i in range(100):
        dt.append(not_main_map(kp1, des1, orb, bf))
    print(f'mean: {sum(dt)/len(dt)}')
    print(f'min: {min(dt)}')
    print(f'max: {max(dt)}')
    # main()
