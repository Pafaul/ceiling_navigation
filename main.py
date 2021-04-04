from sys import argv
from typing import Any, TypeVar
from utils.match_keypoints import match_keypoints
from utils.get_keypoints import get_keypoints_from_image
from utils.camera_calibration import calibrate_camera
from utils.get_position_deltas import get_position_deltas
import cv2
import numpy as np
import time
import configparser

InitialStageInfo = TypeVar('InitialStageInfo', np.array, np.array, cv2.VideoCapture)
MultipleDicts    = TypeVar('MultipleDicts', dict, dict)
MultipleArrays   = TypeVar('MultipleArrays', np.array, np.array)


def main():
    filename = argv[0]
    config, debug = get_config(filename)
    camera_matrix, dist, camera = initial_stage(config)
    descriptor, matcher = initialize_algorithms(config)
    
    tmp_img_current = None
    tmp_des_current, tmp_des_previous = None, None
    tmp_kp_current, tmp_kp_previous = None, None

    delta_R = []
    delta_t = []
    delta_time = []

    if debug:
        current_time = 0
        finish_time = debug['time_to_run']

        while current_time < finish_time:
            res, tmp_img_current = camera.read()
            if not res:
                print('Cannot get image from camera')   
                continue
            
            start_time = time.time()
            tmp_kp_current, tmp_des_current = get_keypoints_from_image(tmp_img_current, descriptor)

            if tmp_kp_previous is not None and tmp_kp_current is not None:
                tmp_pts_current, tmp_pts_previous = match_keypoints(tmp_kp_current, tmp_des_current, tmp_kp_previous, tmp_des_previous)
                R, t = get_position_deltas(tmp_pts_previous, tmp_pts_current, camera_matrix)
                print(R)
                print(t)
                delta_R.append(R.copy())
                delta_t.append(t.copy())
            
            tmp_kp_previous = tmp_kp_current.copy()
            tmp_des_previous = tmp_des_current.copy()
            delta = time.time() - start_time
            delta_time.append(delta)
            current_time += delta


def get_config(filename: str) -> MultipleDicts:
    config = configparser.ConfigParser()
    config.read(filename)
    return config, config['debug']

def initial_stage(config: dict) -> InitialStageInfo:
    mtx, dist = get_camera_matrix(config['camera_matrix'])
    camera = get_camera(config['camera'])
    return [mtx, dist, camera]

def initialize_algorithms(config: dict) -> Any:
    return cv2.SIFT_create(contrastThreshold=0.1), cv2.BFMatcher()

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


def not_main():
    img1 = cv2.imread('1_1.jpg', 0)
    img2 = cv2.imread('1_2.jpg', 0)
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

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    
    camera_matrix = np.array([
        [2.20577589e+03, 0.00000000e+00, 4.01509158e+02],
        [0.00000000e+00, 2.73065460e+03, 3.01560236e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    retval, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC)
    retval, R, t, mask = cv2.recoverPose(retval, pts1, pts2, camera_matrix, mask=mask)

    delta = time.time() - start_time
    print(delta)
    print(F)
    print(R)
    print(t)


if __name__ == '__main__':
    not_main()