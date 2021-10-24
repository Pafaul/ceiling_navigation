import math
import multiprocessing
import time

import numpy as np

from simulation.calculate_keypoint_image_position import calculate_keypoints_on_image
from simulation.camera import Camera
from simulation.camera_movement import BasicCameraMovement
from simulation.plots import show_plots, show_plots_height, show_plots_resolution
from simulation.generate_image import generate_canvas
from simulation.kp_generator import generate_keypoints_equal
from simulation.optical_flow_calculation import get_keypoints_both_pictures, calculate_obj_rotation_matrix, \
    calculate_angles, calculate_angles_delta
from simulation.parse_config import parse_config
from simulation.vizualize_keypoints import draw_keypoints_on_img, show_image, draw_optical_flow


def get_processes_pool(iterations: int):
    cpu_count = multiprocessing.cpu_count()
    processes_to_use = min(
        max(1, cpu_count - 2),
        iterations
    )
    return processes_to_use, multiprocessing.Pool(processes_to_use)


def calculate_camera_fov(camera: Camera) -> list:
    fov = [
        math.atan(camera.fov[0] / 2 / math.pi) * camera.position[2] * 2,
        math.atan(camera.fov[1] / 2 / math.pi) * camera.position[2] * 2,
    ]
    return fov


def run_simulation(
        camera: Camera,
        keypoints: list,
        movement: BasicCameraMovement,
        visualization_config: dict
):
    canvas = generate_canvas(
        height=visualization_config['canvas_size'][0],
        width=visualization_config['canvas_size'][1]
    )

    camera_canvas = generate_canvas(
        height=camera.resolution[0],
        width=camera.resolution[1]
    )

    r = np.eye(3)
    angles = []
    real_angles = []

    current_mask, visible_keypoints, current_kp, = calculate_keypoints_on_image(keypoints, camera)

    for _ in movement.move_camera(camera):
        mask, visible_keypoints, keypoints_px = calculate_keypoints_on_image(keypoints, camera)
        if visualization_config['visualization_enabled']:
            img = draw_keypoints_on_img(canvas, visible_keypoints, visualization_config)
            show_image('camera', img, visualization_config)

        previous_kp = current_kp.copy()
        current_kp = keypoints_px.copy()

        previous_mask = current_mask.copy()
        current_mask = mask.copy()

        prev_img_kp, current_img_kp, real_kp = get_keypoints_both_pictures(keypoints, current_kp, previous_kp,
                                                                           current_mask, previous_mask)
        if visualization_config['visualization_enabled']:
            img = draw_keypoints_on_img(canvas, real_kp, visualization_config)
            show_image('both_pictures', img, visualization_config)

            optical_flow = draw_optical_flow(camera_canvas, previous_mask, current_mask, previous_kp,
                                             current_kp, visualization_config)
            show_image('optical_flow', optical_flow, visualization_config)

        r = calculate_obj_rotation_matrix(
            previous_kp=prev_img_kp,
            current_kp=current_img_kp,
            camera=camera,
            rotation_matrix=r
        )

        real_angles.append(calculate_angles(camera.ufo.rotation_matrix))
        angles.append(calculate_angles(r))

    return real_angles, angles


def simulate_height_change(
        camera: Camera,
        movement: BasicCameraMovement,
        simulation_config: dict,
        visualization_params: dict
) -> tuple:
    camera_initial_position = camera.position.copy()
    camera_initial_rotation = camera.ufo.rotation_matrix.copy()
    heights = np.linspace(
        start=simulation_config['start_height'],
        stop=simulation_config['final_height'],
        num=simulation_config['simulation_points'],
        endpoint=True
    )

    heights_len = len(heights)
    process_count, pool = get_processes_pool(heights_len)
    heights_block = int(heights_len / process_count)
    parameters = []
    for index in range(process_count):
        parameters.append((
            index,
            heights_block,
            heights,
            camera,
            camera_initial_position,
            camera_initial_rotation,
            movement,
            visualization_params,
            simulation_config
        ))
    start_time = time.time()
    return_values = pool.starmap(heights_internal_cycle, parameters)
    print('total time: ', time.time() - start_time)

    real_heights = []
    real_deltas = []
    for rv in return_values:
        real_heights = real_heights + rv[0]
        real_deltas = real_deltas + rv[1]

    return real_heights, real_deltas


def heights_internal_cycle(
        index: int,
        heights_block: int,
        heights: np.array,
        camera: Camera,
        camera_initial_position: np.ndarray,
        camera_initial_rotation: np.ndarray,
        movement: BasicCameraMovement,
        visualization_params: dict,
        simulation_config: dict
):
    deltas = []
    calculated_heights = []
    camera_to_use = Camera(
        camera_initial_position,
        camera_initial_rotation,
        f=camera.f,
        fov=camera.resolution,
        resolution=camera.resolution
    )

    for height in heights[index*heights_block:(index+1)*heights_block]:
        movement_to_use = movement.copy()
        start_time = time.time()
        print('height: ', height)

        new_camera_position = camera_initial_position.copy()
        new_camera_position[2] = height
        camera_to_use.set_camera_position(new_camera_position)
        camera_to_use.set_rotation(camera_initial_rotation)

        fov = calculate_camera_fov(camera_to_use)
        canvas_size = [f * 6 for f in fov]
        visualization_params['coefficients'] = [d / c for (c, d) in
                                                zip(canvas_size, visualization_params['canvas_size'])]
        keypoints = generate_keypoints_equal(np.array(canvas_size), x_keypoint_amount=50, y_keypoint_amount=50)

        real_angles, angles = run_simulation(
            camera=camera_to_use,
            keypoints=keypoints,
            movement=movement_to_use,
            visualization_config=visualization_params
        )

        if simulation_config['plot_enabled']:
            show_plots(real_angles, angles)

        delta = calculate_angles_delta(real_angles, angles)
        deltas.append(delta[-1])
        calculated_heights.append(height)
        # print('Iteration time: {0}, height: {1}'.format(time.time() - start_time, height))
    return calculated_heights, deltas


def simulate_resolution_change(
        camera: Camera,
        movement: BasicCameraMovement,
        simulation_config: dict,
        visualization_params: dict
):
    camera_initial_position = camera.position.copy()
    camera_initial_rotation = camera.ufo.rotation_matrix.copy()
    camera_fov = camera.fov.copy()
    camera_f = camera.f
    resolutions = np.linspace(
        start=simulation_config['initial_resolution'],
        stop=simulation_config['final_resolution'],
        num=simulation_config['simulation_points'],
        endpoint=True
    )

    deltas = []

    for resolution in resolutions:
        start_time = time.time()
        print('resolution: ', resolution)
        camera = Camera(
            initial_position=camera_initial_position.copy(),
            initial_rotation_matrix=camera_initial_rotation.copy(),
            f=camera_f,
            fov=camera_fov.copy(),
            resolution=[int(r) for r in resolution]
        )

        fov = calculate_camera_fov(camera)
        canvas_size = [f * 6 for f in fov]
        visualization_params['coefficients'] = [d / c for (c, d) in
                                                zip(canvas_size, visualization_params['canvas_size'])]
        keypoints = generate_keypoints_equal(np.array(canvas_size), x_keypoint_amount=50, y_keypoint_amount=50)

        real_angles, angles = run_simulation(
            camera=camera,
            keypoints=keypoints,
            movement=movement,
            visualization_config=visualization_params
        )

        if simulation_config['plot_enabled']:
            show_plots(real_angles, angles)

        delta = calculate_angles_delta(real_angles, angles)
        deltas.append(delta[-1])
        print('Iteration time: ', time.time() - start_time)

    return resolutions, deltas


def main():
    camera = None  # type: Camera
    camera, movement, simulation, visualization, plot_config = parse_config('../conf/simulation_config.conf')
    if simulation['type'] == 'default':
        fov = calculate_camera_fov(camera)
        canvas_size = [f * 6 for f in fov]
        visualization['coefficients'] = [c / d for (c, d) in zip(canvas_size, visualization['canvas_size'])]
        keypoints = generate_keypoints_equal(np.array(canvas_size), x_keypoint_amount=50, y_keypoint_amount=50)
        real_angles, angles = run_simulation(
            camera=camera,
            keypoints=keypoints,
            movement=movement,
            visualization_config=visualization
        )
        show_plots(real_angles, angles)

    elif simulation['type'] == 'height':
        height, deltas = simulate_height_change(
            camera=camera,
            movement=movement,
            simulation_config=simulation,
            visualization_params=visualization
        )
        show_plots_height(
            heights=height,
            deltas=deltas,
            simulation_config=simulation,
            movement=movement,
            plot_config=plot_config
        )

    elif simulation['type'] == 'resolution':
        resolutions, deltas = simulate_resolution_change(
            camera=camera,
            movement=movement,
            simulation_config=simulation,
            visualization_params=visualization
        )
        show_plots_resolution(
            resolutions=resolutions,
            deltas=deltas,
            simulation_config=simulation,
            movement=movement,
            plot_config=plot_config
        )


if __name__ == '__main__':
    main()
