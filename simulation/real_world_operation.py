import multiprocessing
import time

import numpy as np

from simulation.calculate_keypoint_image_position import calculate_keypoints_on_image
from simulation.camera import Camera, calculate_camera_fov
from simulation.camera_movement import BasicCameraMovement
from simulation.plots import show_plots, show_plots_height, show_plots_resolution, show_plots_position
from simulation.generate_image import generate_canvas
from simulation.kp_generator import generate_keypoints_equal
from simulation.optical_flow_calculation import get_keypoints_both_pictures, calculate_obj_rotation_matrix, \
    calculate_angles, calculate_angles_delta
from simulation.parse_config import parse_config
from simulation.vizualize_keypoints import draw_keypoints_on_img, show_image, draw_optical_flow


def get_processes_pool(iterations: int):
    cpu_count = multiprocessing.cpu_count()
    processes_to_use = min(
        cpu_count,
        iterations
    )
    return processes_to_use, multiprocessing.Pool(processes_to_use)


def px_to_obj(keypoints: list, mask: list, position: np.ndarray, camera: Camera, rotation_matrix: np.ndarray) -> list:
    r = rotation_matrix.copy()
    u = camera.resolution[0] / 2 * camera.px_mm[0]
    v = camera.resolution[1] / 2 * camera.px_mm[1]
    kp_obj = []
    for (kp, visible) in zip(keypoints, mask):
        if visible:
            obj_position = np.ndarray([3, 1])
            obj_position[2] = 0

            obj_position[0] = position[0] + (obj_position[2] - position[2]) * \
                (r[0, 0] * (kp[0] * camera.px_mm[0] - u) + r[0, 1] * (kp[1] * camera.px_mm[1] - v) - r[0, 2] * camera.f) / \
                (r[2, 0] * (kp[0] * camera.px_mm[0] - u) + r[2, 1] * (kp[1] * camera.px_mm[1] - v) - r[2, 2] * camera.f)

            obj_position[1] = position[1] + (obj_position[2] - position[2]) * \
                (r[1, 0] * (kp[0] * camera.px_mm[0] - u) + r[1, 1] * (kp[1] * camera.px_mm[1] - v) - r[1, 2] * camera.f) / \
                (r[2, 0] * (kp[0] * camera.px_mm[0] - u) + r[2, 1] * (kp[1] * camera.px_mm[1] - v) - r[2, 2] * camera.f)
            kp_obj.append(obj_position)
        else:
            kp_obj.append(np.zeros([3, 1]))

    return kp_obj


def prepare_data_for_lms(keypoints_obj: list, px_keypoints: list, mask: list, r: np.ndarray,
                         z: float, f: float, px_mm: list, resolution: list):
    coefficients = []
    u = resolution[0] / 2 * px_mm[0]
    v = resolution[1] / 2 * px_mm[1]
    for (kp_obj, kp_px, visible) in zip(keypoints_obj, px_keypoints, mask):
        if visible:
            cx1 = (u - kp_px[0] * px_mm[0]) / f
            cy1 = (v - kp_px[1] * px_mm[1]) / f
            cx2 = \
                kp_obj[0] * (r[0, 0] - r[0, 2] * cx1) + \
                kp_obj[1] * (r[1, 0] - r[1, 2] * cx1) - \
                z * (r[2, 0] - r[2, 2] * cx1)
            cy2 = \
                kp_obj[0] * (r[0, 1] - r[0, 2] * cy1) + \
                kp_obj[1] * (r[1, 1] - r[1, 2] * cy1) - \
                z * (r[2, 1] - r[2, 2] * cy1)

            a = r[0, 0] - r[0, 2] * cx1
            b = r[1, 0] - r[1, 2] * cx1
            d = r[0, 1] - r[0, 2] * cy1
            e = r[1, 1] - r[1, 2] * cy1
            coefficients.append((
                a, b, cx2, d, e, cy2
            ))
        else:
            coefficients.append(tuple([0]*6))
    return coefficients


def construct_matrices_lsm(all_coefficients):
    A = np.ndarray([len(all_coefficients) * 2, 2])
    B = np.ndarray([len(all_coefficients) * 2, 1])
    for (coefficients, index) in zip(all_coefficients, range(len(all_coefficients))):
        A[index*2, 0] = coefficients[0]
        A[index*2, 1] = coefficients[1]
        B[index*2, 0] = coefficients[2]
        A[index*2+1, 0] = coefficients[3]
        A[index*2+1, 1] = coefficients[4]
        B[index*2+1, 0] = coefficients[5]

    return A, B


def lsm(A, B):
    return np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(), A)), A.transpose()), B)


def run_calculation(
        camera: Camera,
        video_source
):
    r = np.eye(3)
    angles = []

    calculated_positions = []

    current_mask, visible_keypoints, current_kp, = calculate_keypoints_on_image(keypoints, camera)
    position = np.zeros([3, 1])
    position[2] = camera.position[2]
    current_obj_keypoints = px_to_obj(
        keypoints=current_kp,
        mask=current_mask,
        position=position,
        camera=camera,
        rotation_matrix=r
    )
    total = np.zeros([2, 1])

    for _ in movement.move_camera(camera):
        mask, visible_keypoints, keypoints_px = calculate_keypoints_on_image(keypoints, camera)
        if visualization_config['visualization_enabled']:
            img = draw_keypoints_on_img(canvas, visible_keypoints, visualization_config)
            show_image('camera', img, visualization_config)

        previous_kp = current_kp.copy()
        current_kp = keypoints_px.copy()

        previous_mask = current_mask.copy()
        current_mask = mask.copy()

        prev_img_kp, current_img_kp, real_kp, mask_both_pics = get_keypoints_both_pictures(
            keypoints, current_kp,
            previous_kp,
            current_mask, previous_mask
        )

        r = calculate_obj_rotation_matrix(
            previous_kp=prev_img_kp,
            current_kp=current_img_kp,
            camera=camera,
            rotation_matrix=r
        )

        data_for_lms = prepare_data_for_lms(
            keypoints_obj=current_obj_keypoints,
            px_keypoints=current_kp,
            mask=mask_both_pics,
            r=np.dot(camera.ufo_r_to_camera, r),
            z=camera.position[2],
            f=camera.f,
            px_mm=camera.px_mm,
            resolution=camera.resolution
        )

        A, B = construct_matrices_lsm(data_for_lms)

        X = lsm(A, B)

        position[2] = camera.position[2]

        current_obj_keypoints = px_to_obj(
            keypoints=current_kp,
            mask=current_mask,
            position=position,
            camera=camera,
            rotation_matrix=np.dot(camera.ufo_r_to_camera, r)
        )

        if visualization_config['visualization_enabled']:
            img = draw_keypoints_on_img(canvas, real_kp, visualization_config)
            show_image('both_pictures', img, visualization_config)

            optical_flow = draw_optical_flow(camera_canvas, previous_mask, current_mask, previous_kp,
                                             current_kp, visualization_config)
            show_image('optical_flow', optical_flow, visualization_config)

        real_angles.append(calculate_angles(camera.ufo.rotation_matrix))
        angles.append(calculate_angles(r))
        real_positions.append(camera.position.copy())
        tmp_position = np.zeros([3, 1])
        total = total + X
        tmp_position[0] = total[0]
        tmp_position[1] = total[1]
        tmp_position[2] = camera.position[2]
        calculated_positions.append(tmp_position.copy())

    return real_angles, angles, initial_position, real_positions, calculated_positions


def main():
    camera, movement, simulation, visualization, plot_config = parse_config('../conf/simulation_config.conf')
    if simulation['type'] == 'default':
        real_angles, angles, initial_position, real_positions, calculated_positions = run_simulation(
            camera=camera,
            keypoints=keypoints,
            movement=movement,
            visualization_config=visualization
        )
        show_plots(real_angles, angles, plot_config, simulation, movement)
        show_plots_position(initial_position, real_positions, calculated_positions, movement, plot_config, simulation)


if __name__ == '__main__':
    main()
