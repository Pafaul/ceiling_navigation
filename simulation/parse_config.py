import configparser
import math

import numpy as np

from simulation.camera_movement import BasicCameraMovement, SinMovement, LinearMovement
from simulation.camera import Camera
from simulation.rotation_matrix import calculate_rotation_matrix


def split(string: str, type_to_convert: type):
    string = string.strip()
    if string[0] == '[' and string[-1] == ']':
        string = string[1:-1]
        string = string.replace(' ', '')
        pass
    return [type_to_convert(s) for s in string.split(',')]


def parse_config(filename: str) -> tuple:
    config = create_config_parser(filename)

    camera = parse_camera_config(config)
    movement = parse_movement_parameters(config)
    simulation = parse_simulation_parameters(config)
    visualization = parse_visualization_parameters(config)
    plot_config = parse_plot_config(config)

    return camera, movement, simulation, visualization, plot_config


def create_config_parser(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


def parse_camera_config(config) -> Camera:
    initial_position = np.ndarray([3, 1])
    initial_position_conf = split(config['movement']['initial_position'], float)
    for index in range(3):
        initial_position[index] = initial_position_conf[index]

    initial_angles = split(config['movement']['initial_rotation'], float)
    initial_rotation_matrix = calculate_rotation_matrix(
        initial_angles[0],
        initial_angles[1],
        initial_angles[2]
    )
    camera = Camera(
        initial_position=initial_position,
        initial_rotation_matrix=initial_rotation_matrix,
        f=float(config['camera_parameters']['f']),
        fov=split(config['camera_parameters']['fov'], float),
        resolution=split(config['camera_parameters']['resolution'], int)
    )
    return camera


def parse_simulation_parameters(config) -> dict:
    simulation_config = dict()
    sim_type = config['simulation']['simulation_type']
    simulation_config['type'] = sim_type
    if sim_type == 'default':
         pass

    if sim_type == 'height':
        simulation_config['start_height'] = int(config['simulation_height']['start_height'])
        simulation_config['final_height'] = int(config['simulation_height']['final_height'])
        simulation_config['simulation_points'] = int(config['simulation_height']['simulation_points'])
        simulation_config['plot_enabled'] = config.getboolean('simulation_height', 'plot_enabled')

    if sim_type == 'resolution':
        simulation_config['initial_resolution'] = split(config['simulation_resolution']['initial_resolution'], int)
        simulation_config['final_resolution'] = split(config['simulation_resolution']['final_resolution'], int)
        simulation_config['simulation_points'] = config.getint('simulation_resolution', 'simulation_points')
        simulation_config['plot_enabled'] = config.getboolean('simulation_height', 'plot_enabled')

    return simulation_config


def parse_movement_parameters(config) -> BasicCameraMovement:
    movement_points = int(config['movement']['movement_points'])
    if config['movement']['movement_type'] == 'sin':
        amplitude_x = np.ndarray([3, 1])
        amplitude_w = np.ndarray([3, 1])
        conf_amplitude_x = split(config['movement_sin']['amplitude_x'], float)
        conf_amplitude_w = split(config['movement_sin']['amplitude_w'], float)
        for index in range(3):
            amplitude_x[index] = conf_amplitude_x[index]
            amplitude_w[index] = conf_amplitude_w[index] * math.pi / 180
        movement = SinMovement(
            amplitude_x=amplitude_x,
            amplitude_w=amplitude_w,
            sin_max=math.pi * 2 * float(config['movement_sin']['full_circles']),
            stop_points=movement_points
        )
        return movement
    else:
        start_point = np.ndarray([3, 1])
        finish_point = np.ndarray([3, 1])
        conf_start_point = split(config['movement_linear']['initial_position'], float)
        conf_finish_point = split(config['movement_linear']['final_position'], float)
        for index in range(3):
            start_point[index] = conf_start_point[index]
            finish_point[index] = conf_finish_point[index]

        movement = LinearMovement(
            start_point=start_point,
            finish_point=finish_point,
            initial_angle=split(config['movement_linear']['initial_rotation'], float),
            finish_angle=split(config['movement_linear']['final_rotation'], float),
            stop_points=movement_points
        )
        return movement


def parse_visualization_parameters(config) -> dict:
    visual_config = dict()
    visual_config['visualization_enabled'] = \
        config.getboolean('visualization_parameters', 'visualization_enabled')

    visual_config['visualization_resolution'] = \
        split(config['visualization_parameters']['visualization_resolution'], int)

    visual_config['visualization_pause'] = int(config['visualization_parameters']['visualization_pause'])
    visual_config['canvas_size'] = \
        split(config['visualization_parameters']['canvas_size'], int)

    visual_config['coefficients'] = \
        split(config['visualization_parameters']['coefficients'], float)

    return visual_config


def parse_plot_config(config) -> dict:
    plot_config = dict()
    plot_config['plot_size'] = split(config['plot_config']['plot_size'], int)
    plot_config['dpi'] = config.getint('plot_config', 'dpi')
    plot_config['show_plots'] = config.getboolean('plot_config', 'show_plots')
    plot_config['plot_dir'] = config['plot_config']['plot_dir']
    
    return plot_config
