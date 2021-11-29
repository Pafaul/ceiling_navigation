import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

from camera_navigation.calculations import calculate_obj_rotation_matrix, px_to_obj, \
    prepare_data_for_lsm, construct_matrices_lsm, lsm, calculate_angles, calculate_optical_flow, calculate_of_parameters
from camera_navigation.dense_of import dense_of_loop
from camera_navigation.detectors import BasicDetector, SIFTDetector, ORBDetector
from camera_navigation.matchers import BasicMatcher, BFFMatcher
from camera_navigation.plot import DynamicPlotUpdate
from camera_navigation.video_sources import BasicVideoSource, VideoCaptureSource
from camera_navigation.visualization import optical_flow_visualization


def get_video_source(config: dict) -> BasicVideoSource:
    return VideoCaptureSource(config)


def main():
    config = dict()
    config['video_source'] = '../videos/good_conditions.avi'

    video_source = get_video_source(config)
    init_camera_internal_matrix = np.array([
        [910.55399, 0., 643.59702],
        [0., 908.73393, 374.39075],
        [0., 0., 1.]
    ])
    dense_of_loop(
        video_source=video_source,
        camera_matrix=init_camera_internal_matrix
    )


if __name__ == '__main__':
    main()
