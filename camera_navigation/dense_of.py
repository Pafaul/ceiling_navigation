import cv2
import matplotlib.pyplot as plt
import numpy as np

from camera_navigation.calculations import calculate_optical_flow, calculate_obj_rotation_matrix, calculate_angles, \
    filter_optical_flow, calculate_of_parameters
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
            kp_x = x_center + int(max_loc[0] - search_window_size[0] / 2)
            kp_y = y_center + int(max_loc[1] - search_window_size[1] / 2)
            of_curr.append((kp_x, kp_y))

    of_prev = np.int32(of_prev)
    of_curr = np.int32(of_curr)
    return of_prev, of_curr


def dense_of_loop(video_source: BasicVideoSource, camera_matrix: np.ndarray):
    prev_img = None
    current_img = None

    r = np.eye(3)
    plt.ion()

    of_plot_orig = DynamicPlotUpdate(subplots=2, marker_type='o')
    of_plot_filtered = DynamicPlotUpdate(subplots=2, marker_type='o')
    angle_plots = DynamicPlotUpdate(subplots=3, marker_type='-')

    wx = []
    wy = []
    wz = []

    for (result, frame) in video_source.get_frame():
        if result:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (int(w / 1.5), int(h / 1.5)))
            grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if current_img is not None:
                prev_img = current_img.copy()

            current_img = grayscale_img.copy()

            if current_img is not None and prev_img is not None:
                kp_prev, kp_curr = dense_optical_flow(
                    previous_img=prev_img,
                    current_img=current_img,
                    template_window_size=[32, 32],
                    search_window_size=[64, 64],
                    x_regions=7,
                    y_regions=7
                )

                vectors = calculate_optical_flow(kp_curr, kp_prev)
                a, phi = calculate_of_parameters(vectors)
                of_plot_orig.update_data([
                    [list(range(len(a))), a],
                    [list(range(len(phi))), phi]
                ])

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
                    a, phi = calculate_of_parameters(res_of)
                    of_plot_filtered.update_data([
                        [list(range(len(a))), a],
                        [list(range(len(phi))), phi]
                    ])

                    of_img = optical_flow_visualization(prev_img, kp_prev, res_of)
                    of_img = draw_kp(of_img, kp_curr)
                    cv2.imshow('of', of_img)
                    cv2.waitKey(1)

                    tmp_r = calculate_obj_rotation_matrix(
                        current_kp=kp_curr,
                        previous_kp=kp_prev,
                        camera_internal_matrix=camera_matrix,
                        rotation_matrix=r
                    )
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
                else:
                    current_img = None
                    prev_img = None


def draw_kp(img: np.ndarray, kp):
    for curr_kp in kp:
        cv2.drawMarker(
            img=img,
            position=(curr_kp[0], curr_kp[1]),
            color=(255,),
            markerType=cv2.MARKER_TRIANGLE_DOWN
        )
    return img
