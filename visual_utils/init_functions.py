from kalman.kalman_filtering import KalmanFiltering
from filterpy.kalman.kalman_filter import KalmanFilter
import numpy as np
import configparser
import cv2
from typing import Any, TypeVar
from camera_utils.camera_distance_calculator import CameraDistanceCalculator
from camera_utils.camera_calibration import calibrate_camera

InitialStageInfo = TypeVar('InitialStageInfo', np.array, np.array, cv2.VideoCapture, CameraDistanceCalculator)
MultipleDicts    = TypeVar('MultipleDicts', dict, dict)
MultipleArrays   = TypeVar('MultipleArrays', np.array, np.array)

def get_config(filename: str) -> MultipleDicts:
    config = configparser.ConfigParser()
    config.read(filename)
    return config, config['debug']

def initial_stage(config: dict) -> InitialStageInfo:
    mtx, dist = get_camera_matrix(config['camera_matrix'])
    camera = get_camera(config['camera'])
    camera_distance_calculator = CameraDistanceCalculator(
        config['camera_parameters']['h'], 
        config['camera_parameters']['alphaX'], 
        config['camera_parameters']['alphaY'],
        config['camera_parameters']['px'],
        config['camera_parameters']['py'],  
    )
    return [mtx, dist, camera, camera_distance_calculator]

def initialize_algorithms(config: dict) -> Any:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    fbm = cv2.FlannBasedMatcher(index_params, search_params)
    return cv2.SIFT_create(contrastThreshold=0.1), cv2.BFMatcher()

def get_camera_matrix(config: dict) -> list[np.array]:
    if config['calibration_image'] != '':
        return calibrate_camera(calibration_image=config['calibration_image'])
    else:
        return calibrate_camera(calibration_file=config['calibration_file'])

def get_camera(config: dict) -> cv2.VideoCapture:
    if config['source'] != '':
        source = config['source']
        source = int(source) if source.isdigit() else source
        camera = cv2.VideoCapture(source)
        return camera
    else:
        raise Exception('Video source is not specified')

def initialize_kalman_filter(x, dim_z, p_x, r_x):
    return KalmanFiltering(x=x, dim_z=dim_z, p_x=p_x, r_x=r_x)

def get_thresholds(config):
    threshold_config = dict(config['vector_thresholds'])
    threshold_config['angle_threshold'] = float(threshold_config['angle_threshold'])
    threshold_config['dist_threshold'] = float(threshold_config['dist_threshold'])
    threshold_config['matcher_threshold'] = int(threshold_config['matcher_threshold'])
    return threshold_config
