import cv2
import numpy as np
from matplotlib import pyplot as plt

from camera_navigation.calculations import calculate_angles, calculate_obj_rotation_matrix, calculate_of_parameters, \
    calculate_optical_flow
from camera_navigation.detectors import ORBDetector
from camera_navigation.main import get_video_source
from camera_navigation.matchers import BFFMatcher
from camera_navigation.visualization import optical_flow_visualization
from test import DynamicPlotUpdate


def kp_of():
    config = dict()
    config['video_source'] = '../videos/good_conditions.avi'

    video_source = get_video_source(config)

    current_kp = []
    current_des = []

    detector = ORBDetector()
    matcher = BFFMatcher(detector=detector)

    r = np.eye(3)
    init_camera_internal_matrix = np.array([
        [910.55399, 0., 643.59702],
        [0., 908.73393, 374.39075],
        [0., 0., 1.]
    ])

    distortion = np.array([0.096794, -0.209042, 0.003097, -0.005152, 0.000000])

    previous_img = None

    plt.ion()
    of_plot = DynamicPlotUpdate(subplots=2, marker_type='o')
    angle_plots = DynamicPlotUpdate(subplots=3, marker_type='-')

    wx = []
    wy = []
    wz = []

    frame_skipped = False
    for (result, frame) in video_source.get_frame():
        if result:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w / 1.5), int(h / 1.5)))
            # # frame = cv2.bilateralFilter(frame, 9, 75, 75, cv2.BORDER_DEFAULT)
            # kernel = np.array([[1.0, 1.0, 1.0],
            #                    [1.0, 1.0, 1.0],
            #                    [1.0, 1.0, 1.0]])
            #
            # kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
            # frame = cv2.filter2D(frame, -1, kernel)
            #
            kernel = np.array([[0.0, -1.0, 0.0],
                               [-1.0, 5.0, -1.0],
                               [0.0, -1.0, 0.0]])

            kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)
            frame = cv2.filter2D(frame, -1, kernel)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)

            # calc_camera_internal_matrix, roi = cv2.getOptimalNewCameraMatrix(
            #     cameraMatrix=init_camera_internal_matrix,
            #     distCoeffs=distortion,
            #     imageSize=[w, h],
            #     alpha=0,
            #     newImgSize=[w, h]
            # )
            #
            # dst = cv2.undistort(frame, init_camera_internal_matrix, distortion, None, calc_camera_internal_matrix)
            # x, y, w, h = roi
            # dst = dst[y:y + h, x:x + w]
            # cv2.imshow('undistorted', dst)
            # cv2.waitKey(1)
            calc_camera_internal_matrix = init_camera_internal_matrix.copy()
            dst = frame.copy()

            if not frame_skipped:
                previous_kp = current_kp
                previous_des = current_des
            else:
                print('Frame skipped')
            frame_skipped = False

            grayscale_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray', grayscale_img)
            cv2.waitKey(1)
            current_kp, current_des = detector.get_keypoints(grayscale_img)

            current_kp_img = cv2.drawKeypoints(dst.copy(), current_kp, dst.copy(),
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('Current keypoints', current_kp_img)
            cv2.waitKey(1)

            if len(current_kp) > 10 and len(previous_kp) > 10:
                kp_1, kp_2 = matcher.match_keypoints(
                    kp_1=current_kp,
                    kp_2=previous_kp,
                    des_1=current_des,
                    des_2=previous_des
                )
                if len(kp_1) > 8 and len(kp_2) > 8:
                    vectors = calculate_optical_flow(kp_1, kp_2)
                    of_img = optical_flow_visualization(previous_img, kp_2, vectors)
                    cv2.imshow('of', of_img)
                    cv2.waitKey(1)
                    a, phi = calculate_of_parameters(vectors)
                    # a.sort()
                    # phi.sort()

                    of_plot.update_data([
                        [list(range(len(a))), a],
                        [list(range(len(phi))), phi]
                    ])

                    current_matches_img = cv2.drawMatches(
                        img1=frame.copy(),
                        keypoints1=current_kp,
                        img2=previous_img.copy(),
                        keypoints2=previous_kp,
                        matches1to2=matcher.get_raw_matches(
                            kp_1=current_kp,
                            kp_2=previous_kp,
                            des_1=current_des,
                            des_2=previous_des
                        ),
                        outImg=frame.copy()
                    )
                    cv2.imshow('matches', cv2.resize(current_matches_img, (1280, 360)))
                    cv2.waitKey(1)

                    tmp_r = calculate_obj_rotation_matrix(
                        current_kp=kp_1,
                        previous_kp=kp_2,
                        camera_internal_matrix=init_camera_internal_matrix,
                        rotation_matrix=r
                    )

                    # resolution = [frame.shape[1], frame.shape[0]]
                    #
                    # obj_points = px_to_obj(
                    #     keypoints=kp_2,
                    #     camera_height=camera_height,
                    #     f=f,
                    #     resolution=resolution,
                    #     px_mm=px_mm,
                    #     rotation_matrix=r
                    # )
                    #
                    # data_for_lms = prepare_data_for_lsm(
                    #     keypoints_obj=obj_points,
                    #     px_keypoints=kp_1,
                    #     r=tmp_r,
                    #     dz=camera_height - camera_height,
                    #     f=f,
                    #     px_mm=px_mm,
                    #     resolution=resolution
                    # )
                    #
                    # A, B = construct_matrices_lsm(data_for_lms)
                    #
                    # X = lsm(A, B)
                    #
                    # final_position = final_position + X

                    r = tmp_r.copy()
                    calculated_angles = calculate_angles(r)
                    wx.append(calculated_angles[0])
                    wy.append(calculated_angles[1])
                    wz.append(calculated_angles[2])
                    print(f'angles: {calculated_angles}')

                    angle_plots.update_data([
                        [list(range(len(wx))), wx],
                        [list(range(len(wy))), wy],
                        [list(range(len(wz))), wz],
                    ])
                    # print(f'position: {final_position}')
                else:
                    frame_skipped = True
            else:
                # TODO: skip frame
                pass
        else:
            print('Frame skipped')

        if frame is not None:
            previous_img = frame.copy()
        else:
            break

    pass