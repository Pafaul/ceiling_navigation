import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera_navigation.calculations import calculate_optical_flow, calculate_obj_rotation_matrix, calculate_angles, \
    filter_optical_flow, calculate_of_parameters, px_to_obj, prepare_data_for_lsm, construct_matrices_lsm, lsm, \
    px_to_obj_2, prepare_data_for_lsm_2, construct_matrices_lsm_2
from camera_navigation.plot import DynamicPlotUpdate
from camera_navigation.video_sources import BasicVideoSource
from camera_navigation.visualization import optical_flow_visualization


def dense_optical_flow(
        previous_img: np.ndarray,
        current_img: np.ndarray,
        template_window_size: list,
        search_window_size: list,
        x_regions: int = 4,
        y_regions: int = 4,
):
    h, w = previous_img.shape[:2]
    delta_x = w / x_regions
    delta_y = h / y_regions

    of_prev = []
    of_curr = []
    for x_index in range(0, x_regions):
        for y_index in range(0, y_regions):
            x_center, y_center = int(delta_x * (x_index + 1/2)), int(delta_y * (y_index + 1/2))
            of_prev.append((x_center, y_center))
            template_img = previous_img[
                int(y_center - template_window_size[1] / 2):int(y_center + template_window_size[1] / 2),
                int(x_center - template_window_size[0] / 2):int(x_center + template_window_size[0] / 2)
            ]
            search_region = current_img[
                int(y_center - search_window_size[1] / 2):int(y_center + search_window_size[1] / 2),
                int(x_center - search_window_size[0] / 2):int(x_center + search_window_size[0] / 2)
            ]

            res = cv2.matchTemplate(search_region, template_img, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            kp_x = x_center + int(max_loc[0] - int(res.shape[0] / 2))
            kp_y = y_center + int(max_loc[1] - int(res.shape[1] / 2))
            of_curr.append((kp_x, kp_y))

    of_prev = np.int32(of_prev)
    of_curr = np.int32(of_curr)
    return of_prev, of_curr


def draw_regions(img: np.ndarray, template_window_size: list, search_window_size: list, x_regions: int, y_regions: int):
    h, w = img.shape[:2]
    delta_x = w / x_regions
    delta_y = h / y_regions
    for x_index in range(0, x_regions):
        for y_index in range(0, y_regions):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255,)
            thickness = 1
            line_type = 1

            x_center, y_center = int(delta_x * (x_index + 1/2)), int(delta_y * (y_index + 1/2))
            cv2.rectangle(
                img=img,
                pt1=(int(x_center - template_window_size[0]/2), int(y_center - template_window_size[1] / 2)),
                pt2=(int(x_center + template_window_size[0]/2), int(y_center + template_window_size[1] / 2)),
                color=(255,),
                thickness=3
            )

            cv2.rectangle(
                img=img,
                pt1=(int(x_center - search_window_size[0] / 2), int(y_center - search_window_size[1] / 2)),
                pt2=(int(x_center + search_window_size[0] / 2), int(y_center + search_window_size[1] / 2)),
                color=(255,),
                thickness=3
            )

            cv2.putText(
                img=img,
                org=(int(x_center - template_window_size[0]/2) + 7, int(y_center + template_window_size[1] / 2) - 7),
                fontFace=font,
                fontScale=font_scale,
                color=font_color,
                thickness=thickness,
                lineType=line_type,
                text=f'T{x_index*x_regions + y_index + 1}'
            )

            cv2.putText(
                img=img,
                org=(int(x_center - search_window_size[0] / 2 + search_window_size[0] / 3), int(y_center + search_window_size[1] / 2) - 7),
                fontFace=font,
                fontScale=font_scale,
                color=font_color,
                thickness=thickness,
                lineType=line_type,
                text=f'RI{x_index * x_regions + y_index + 1}'
            )

    return img


def dense_of_loop(video_source: BasicVideoSource, camera_matrix: np.ndarray):
    prev_img = None
    current_img = None

    r = np.eye(3)
    plt.ion()

    camera_height = 1.2

    final_position = np.zeros([3, 1])
    x_pos, y_pos = [], []

    wx = []
    wy = []
    wz = []

    frame_skipped = False
    frame_num = 0

    start_time = time.time()
    for (result, frame) in video_source.get_frame():
        if result:
            frame_num += 1
            print(frame_num)
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (1080, 720)) # (int(w / 1.5), int(h / 1.5)))
            grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if current_img is not None and not frame_skipped:
                prev_img = current_img.copy()
            frame_skipped = False
            current_img = grayscale_img.copy()

            if current_img is not None and prev_img is not None:
                kp_prev, kp_curr = dense_optical_flow(
                    previous_img=prev_img,
                    current_img=current_img,
                    template_window_size=[50, 50],
                    search_window_size=[96, 96],
                    x_regions=7,
                    y_regions=6
                )

                vectors = calculate_optical_flow(kp_curr, kp_prev)
                a, phi = calculate_of_parameters(vectors)

                mean_a = sum(a) / len(a)
                if mean_a < 5:
                    frame_skipped = True
                    continue

                res_mask, res_of = filter_optical_flow(vectors)

                tmp_kp_curr = []
                tmp_kp_prev = []

                for index in range(len(res_mask)):
                    if res_mask[index]:
                        tmp_kp_prev.append(kp_prev[index])
                        tmp_kp_curr.append(kp_curr[index])

                kp_curr = np.int32(tmp_kp_curr)
                kp_prev = np.int32(tmp_kp_prev)

                if len(res_of) > 8:
                    obj_points = px_to_obj_2(
                        keypoints=kp_prev,
                        r=r,
                        position=final_position,
                        camera_matrix=camera_matrix,
                        h=camera_height
                    )

                    tmp_r = calculate_obj_rotation_matrix(
                        current_kp=kp_curr,
                        previous_kp=kp_prev,
                        camera_internal_matrix=camera_matrix,
                        rotation_matrix=r
                    )
                    r = tmp_r.copy()

                    coefficients = prepare_data_for_lsm_2(
                        keypoints=kp_curr,
                        keypoints_obj=obj_points,
                        r=r,
                        camera_matrix=camera_matrix,
                        h=camera_height
                    )

                    A, B = construct_matrices_lsm_2(coefficients)

                    X = lsm(A, B)

                    X_tmp = [float(X[i][0]) if abs(float(X[i][0])) > 0.01 else 0 for i in range(3)]
                    X[0] = X_tmp[0]
                    X[1] = X_tmp[1]
                    X[2] = X_tmp[2]

                    final_position += X
                    print(f'pos: {list(final_position)}')
                    x_pos.append(float(final_position[0]))
                    y_pos.append(float(final_position[1]))

                    calculated_angles = calculate_angles(r)
                    wx.append(calculated_angles[0])
                    wy.append(calculated_angles[1])
                    wz.append(calculated_angles[2])
                    print(f'angles: {calculated_angles}')
                else:
                    current_img = None
                    prev_img = None
        else:
            break

    finish_time = time.time()
    fps = frame_num/(finish_time - start_time)
    print(f'fps: {fps}')

    plt.ioff()
    fig, ax = plt.subplots()
    ax.plot(x_pos, y_pos, linewidth=4.0)
    ax.grid()
    plt.xlabel('Смещение по оси X, м')
    plt.ylabel('Смещение по оси Y, м')
    plt.show()
    #
    print(f'fin pos: {x_pos[-1]}, {y_pos[-1]}')
    #
    fig, ax = plt.subplots()
    ax.plot(list(range(len(wx))), wx, linewidth=4.0)
    ax.grid()
    plt.xlabel('Номер кадра')
    plt.ylabel('Рассчитанное значение угла wx, градусы')
    plt.show()
    print(f'fin wx: {wx[-1]}')
    wx = [abs(w) for w in wx]
    print(f'max wx: {max(wx)}')
    #
    fig, ax = plt.subplots()
    ax.plot(list(range(len(wy))), wy, linewidth=4.0)
    ax.grid()
    plt.xlabel('Номер кадра')
    plt.ylabel('Рассчитанное значение угла wy, градусы')
    plt.show()
    print(f'fin wy: {wy[-1]}')
    wy = [abs(w) for w in wy]
    print(f'max wy: {max(wy)}')
    #
    fig, ax = plt.subplots()
    ax.plot(list(range(len(wz))), wz, linewidth=4.0)
    ax.grid()
    plt.xlabel('Номер кадра')
    plt.ylabel('Рассчитанное значение угла wz, градусы')
    plt.show()
    print(f'fin wz: {wz[-1]}')


def draw_kp(img: np.ndarray, kp):
    for curr_kp in kp:
        cv2.drawMarker(
            img=img,
            position=(curr_kp[0], curr_kp[1]),
            color=(255,),
            markerType=cv2.MARKER_TRIANGLE_DOWN
        )
    return img
