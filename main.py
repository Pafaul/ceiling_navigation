from lsm.least_square_method import map_lsm, seq_lsm
from sys import argv
from typing import Any, TypeVar
import cv2
import numpy as np
import time
import configparser
from math import pi, atan2
import matplotlib.pyplot as plt

from camera_utils.match_keypoints import match_keypoints
from camera_utils.get_keypoints import get_keypoints_from_image
from camera_utils.camera_calibration import calibrate_camera
from camera_utils.get_position_deltas import get_position_deltas
from kalman import KalmanFiltering


InitialStageInfo = TypeVar('InitialStageInfo', np.array, np.array, cv2.VideoCapture)
MultipleDicts    = TypeVar('MultipleDicts', dict, dict)
MultipleArrays   = TypeVar('MultipleArrays', np.array, np.array)

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


def main():
    filename = argv[1]
    config, debug = get_config(filename)
    camera_matrix, dist, camera = initial_stage(config)
    descriptor, matcher = initialize_algorithms(config)

    tmp_img_current = None
    tmp_des_current, tmp_des_previous = None, None
    tmp_kp_current, tmp_kp_previous = None, None

    Z = []
    errors = []
    delta_time = []

    if debug['dbg_mode']:
        current_time = 0
        finish_time = float(debug['time_to_run'])
        start_time = time.time()

        i = 0
        while i < finish_time:
            res, tmp_img_current = camera.read()
            tmp_img_current = cv2.cvtColor(tmp_img_current, cv2.COLOR_BGR2GRAY)
            cv2.imshow('cap', tmp_img_current)
            cv2.waitKey(1)
            dt = time.time() - start_time
            if not res or tmp_img_current is None or len(tmp_img_current) == 0:
                print('Cannot get image from camera')   
                time.sleep(0.1)
                continue

            start_time = time.time()
            tmp_kp_current, tmp_des_current = get_keypoints_from_image(tmp_img_current, descriptor)

            if tmp_kp_previous is not None and tmp_kp_current is not None:
                tmp_pts_current, tmp_pts_previous = match_keypoints(tmp_kp_current, tmp_des_current, tmp_kp_previous, tmp_des_previous, matcher)
                current_measures, measures_error = get_position_deltas(tmp_pts_previous, tmp_pts_current, camera_matrix)
                if len(current_measures) == 0:
                    continue
                current_measures[6] = dt
                Z.append(current_measures.copy())
                errors.append(measures_error.copy())
                i += 1
                print(f'Image: {i}')

            tmp_kp_previous = tmp_kp_current.copy()
            tmp_des_previous = tmp_des_current.copy()
            delta_time.append(dt)
            current_time += dt

        for index in range(7):
            plt.plot([t for t in range(len(Z))], [y[index] for y in Z])
            plt.xlabel('Номер шага')
            plt.ylabel(measurments_descriptions[index])
            plt.show()

        for index in range(7):
            plt.plot([t for t in range(len(Z))], [y[index] for y in errors])
            plt.xlabel('Номер шага')
            plt.ylabel(error_descriptions[index])
            plt.show()
        


def get_config(filename: str) -> MultipleDicts:
    config = configparser.ConfigParser()
    config.read(filename)
    return config, config['debug']

def initial_stage(config: dict) -> InitialStageInfo:
    mtx, dist = get_camera_matrix(config['camera_matrix'])
    camera = get_camera(config['camera'])
    return [mtx, dist, camera]

def initialize_algorithms(config: dict) -> Any:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    return cv2.SIFT_create(contrastThreshold=0.1), cv2.FlannBasedMatcher(index_params, search_params)

def get_camera_matrix(config: dict) -> list[np.array]:
    if config['calibration_image'] != '':
        return calibrate_camera(calibration_image=config['calibration_image'])
    else:
        return calibrate_camera(calibration_file=config['calibration_file'])

def get_camera(config: dict) -> cv2.VideoCapture:
    if config['source'] != '':
        camera = cv2.VideoCapture(config['source'])
        return camera
    else:
        raise Exception('Video source is not specified')


def not_main_seq():
    img1 = cv2.imread('./images/1_2.jpg', 0)
    img2 = cv2.imread('./images/1_1.jpg', 0)
    #orb = cv2.ORB_create()
    orb = cv2.SIFT_create(contrastThreshold=0.1)
    #orb = cv2.SIFT_create()
    # BFMatcher with default params
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # bf = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher()

    kp1, des1 = orb.detectAndCompute(img1, None)
    start_time = time.time()
    kp2, des2 = orb.detectAndCompute(img2, None)

    
    
    matches = bf.knnMatch(des1,des2,k=2)
    
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
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
    angle_y = atan2(-R[2][0], (R[2][1]**2 + R[2][2]**2)**0.5)
    angle_z = atan2(R[1][0], R[0][0])
    toDeg = lambda x: x*180/pi
    delta = time.time() - start_time
    print(f'time of execution: {delta}')
    print(f'delta in pixels: {X}')
    print(f'KP ok: {mask.count(True)}')
    print(f'angles:\nox: {toDeg(angle_x)}, oy: {toDeg(angle_y)}, oz: {toDeg(angle_z)}')
    print(error_EV)
    print(error_DP)

def not_main_map():
    img1 = cv2.imread('./images/1.jpg', 0)
    img2 = cv2.imread('./images/img_part.jpg', 0)
    #orb = cv2.ORB_create()
    orb = cv2.SIFT_create(contrastThreshold=0.1)
    #orb = cv2.SIFT_create()

    
    kp1, des1 = orb.detectAndCompute(img1, None)
    start_time = time.time()
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    pts1 = []
    pts2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    X, mask, error_EV, error_DP = map_lsm(pts1, pts2)

    pts1 = np.int32(pts1[mask])
    pts2 = np.int32(pts2[mask])

    X, mask, error_EV, error_DP = map_lsm(pts1, pts2)

    delta = time.time() - start_time
    toDeg = lambda x: x*180/pi
    print(f'time of execution: {delta}')
    print(f'delta in pixels: {X[2:]}')
    print(f'KP ok: {mask.count(True)}')
    print(f'angles:\noz: {toDeg(atan2(-X[1], X[0]))}')
    print(error_EV)
    print(error_DP)

if __name__ == '__main__':
    not_main_seq()
    #main()