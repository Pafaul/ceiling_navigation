import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from simulation.camera_movement import SinMovement


def show_plots(real_angles, angles, plot_config, simulation_config, movement):
    blue_patch = mpatches.Patch(color='blue', label='Действительный угол')
    red_patch = mpatches.Patch(color='red', label='Рассчитанный угол wx')
    green_patch = mpatches.Patch(color='green', label='Рассчитанный угол wy')
    purple_patch = mpatches.Patch(color='purple', label='Рассчитанный угол wz')

    plot_size = plot_config['plot_size']
    dpi = plot_config['dpi']
    fig_size = tuple([pix / dpi for pix in plot_size])

    file_templates = [
        '{0}/wx_{1}_{2}_{3}.png',
        '{0}/wy_{1}_{2}_{3}.png',
        '{0}/wz_{1}_{2}_{3}.png',
        '{0}/wd_{1}_{2}_{3}.png',
    ]

    if isinstance(movement, SinMovement):
        file_name_movement = list([str(round(float(x))) for x in movement.amplitude_x])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.amplitude_w])
        file_name_movement = 'sin_' + '_'.join(file_name_movement)
    else:
        file_name_movement = list([str(round(float(x))) for x in movement.moving_deltas[0]])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.rotation_deltas[0]])
        file_name_movement = 'linear_' + '_'.join(file_name_movement)

    plt.figure(figsize=fig_size, dpi=dpi)
    plt.grid(True)
    plt.plot([angle[0] for angle in real_angles], 'b', linewidth=3)
    plt.plot([-angle[0] for angle in angles], 'r', linewidth=3)
    plt.xlabel('Итерация')
    plt.ylabel('Угол, градусы')
    plt.legend(handles=[blue_patch, red_patch], loc='upper right')
    plt.savefig(
        file_templates[0].format(
            plot_config['plot_dir'],
            simulation_config['type'],
            movement.movement_points,
            file_name_movement
        ),
        bbox_inches='tight'
    )
    if plot_config['show_plots']:
        plt.show()

    plt.figure(figsize=fig_size, dpi=dpi)
    plt.grid(True)
    plt.plot([angle[1] for angle in real_angles], 'b', linewidth=3)
    plt.plot([-angle[1] for angle in angles], 'g', linewidth=3)
    plt.xlabel('Итерация')
    plt.ylabel('Угол, градусы')
    plt.legend(handles=[blue_patch, green_patch], loc='upper right')
    plt.savefig(
        file_templates[1].format(
            plot_config['plot_dir'],
            simulation_config['type'],
            movement.movement_points,
            file_name_movement
        ),
        bbox_inches='tight'
    )
    if plot_config['show_plots']:
        plt.show()

    plt.figure(figsize=fig_size, dpi=dpi)
    plt.grid(True)
    plt.plot([angle[2] for angle in real_angles], 'b', linewidth=3)
    plt.plot([angle[2] for angle in angles], 'purple', linewidth=3)
    plt.xlabel('Итерация')
    plt.ylabel('Угол, градусы')
    plt.legend(handles=[blue_patch, purple_patch], loc='upper right')
    plt.savefig(
        file_templates[2].format(
            plot_config['plot_dir'],
            simulation_config['type'],
            movement.movement_points,
            file_name_movement
        ),
        bbox_inches='tight'
    )
    if plot_config['show_plots']:
        plt.show()

    red_patch = mpatches.Patch(color='red', label='Ошибка wx')
    green_patch = mpatches.Patch(color='green', label='Ошибка wy')
    purple_patch = mpatches.Patch(color='purple', label='Ошибка wz')

    dx = [real_angle[0] + angle[0] for (real_angle, angle) in zip(real_angles, angles)]
    dy = [real_angle[1] + angle[1] for (real_angle, angle) in zip(real_angles, angles)]
    dz = [real_angle[2] - angle[2] for (real_angle, angle) in zip(real_angles, angles)]
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.grid(True)
    plt.plot(dx, 'r', linewidth=3)
    plt.plot(dy, 'g', linewidth=3)
    plt.plot(dz, 'purple', linewidth=3)
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка определения угла, градусы')
    plt.legend(handles=[red_patch, green_patch, purple_patch], loc='upper right')
    plt.savefig(
        file_templates[3].format(
            plot_config['plot_dir'],
            simulation_config['type'],
            movement.movement_points,
            file_name_movement
        ),
        bbox_inches='tight'
    )
    if plot_config['show_plots']:
        plt.show()

    print(f'angles fin: ({[angle[0] for angle in angles][-1]}, '
          f'{[angle[1] for angle in angles][-1]}, '
          f'{[angle[2] for angle in angles][-1]})')
    print(f'angles delta: ({dx[-1]}, {dy[-1]}, {dz[-1]})')
    dx = [abs(d) for d in dx]
    dy = [abs(d) for d in dy]
    dz = [abs(d) for d in dz]
    print(f'max delta: ({max(dx)}, {max(dy)}, {max(dz)})')


def show_plots_height(heights, deltas, simulation_config, movement, plot_config):
    red_patch = mpatches.Patch(color='red', label='Конечная ошибка угла wx')
    green_patch = mpatches.Patch(color='green', label='Конечная ошибка угла wy')
    purple_patch = mpatches.Patch(color='purple', label='Конечная ошибка угла wz')

    plot_size = plot_config['plot_size']
    dpi = plot_config['dpi']
    fig_size = tuple([pix / dpi for pix in plot_size])

    file_templates = [
        '{0}/wx_{1}_{2}_{3}.png',
        '{0}/wy_{1}_{2}_{3}.png',
        '{0}/wz_{1}_{2}_{3}.png'
    ]

    patches = [
        red_patch,
        green_patch,
        purple_patch
    ]

    colors = ['r', 'g', 'purple']

    if isinstance(movement, SinMovement):
        file_name_movement = list([str(round(float(x))) for x in movement.amplitude_x])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.amplitude_w])
        file_name_movement = 'sin_' + '_'.join(file_name_movement)
    else:
        file_name_movement = list([str(round(float(x))) for x in movement.moving_deltas[0]])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.rotation_deltas[0]])
        file_name_movement = 'linear_' + '_'.join(file_name_movement)

    axis_value = ['x', 'y', 'z']
    for axis_index in range(0, 3):
        mean = []
        std = []
        heights_val = []
        for index in range(0, int(len(heights)/100)):
            mean.append(np.mean([delta[axis_index] for delta in deltas][index*100: (index+1)*100]))
            std.append(np.std([delta[axis_index] for delta in deltas][index*100: (index+1)*100]))
            heights_val.append(f'{(index+1)*100}')

        mean_file_template = '{0}/w{1}_{2}_{3}_{4}_mean.png'
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot(heights_val, mean, linewidth=3)
        plt.xlabel('Высоты, м')
        plt.ylabel(f'Среднее значение ошибки угла w{axis_value[axis_index]}, градусы')
        plt.savefig(
            mean_file_template.format(
                plot_config['plot_dir'],
                axis_value[axis_index],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        plt.show()

        std_file_template = '{0}/w{1}_{2}_{3}_{4}_std.png'
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot(heights_val, std, linewidth=3)
        plt.xlabel('Высоты, м')
        plt.ylabel(f'Значение СКО ошибки угла w{axis_value[axis_index]}, градусы')
        plt.savefig(
            std_file_template.format(
                plot_config['plot_dir'],
                axis_value[axis_index],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        plt.show()

    for (index, file_name_template, patch, color) in zip(
            list(range(len(file_templates))), file_templates, patches, colors
    ):
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot(heights, [delta[index] for delta in deltas], color, linewidth=3)
        plt.xlabel('Высота, м')
        plt.ylabel('Ошибка, градусы')
        plt.legend(handles=[patch], loc='upper right')
        plt.savefig(
            file_name_template.format(
                plot_config['plot_dir'],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        if plot_config['show_plots']:
            plt.show()


def show_plots_resolution(resolutions, deltas, simulation_config, movement, plot_config):
    red_patch = mpatches.Patch(color='red', label='Конечная ошибка угла wx')
    green_patch = mpatches.Patch(color='green', label='Конечная ошибка угла wy')
    purple_patch = mpatches.Patch(color='purple', label='Конечная ошибка угла wz')

    plot_size = plot_config['plot_size']
    dpi = plot_config['dpi']
    fig_size = tuple([pix / dpi for pix in plot_size])

    file_templates = [
        '{0}/wx_{1}_{2}_{3}.png',
        '{0}/wy_{1}_{2}_{3}.png',
        '{0}/wz_{1}_{2}_{3}.png'
    ]

    patches = [
        red_patch,
        green_patch,
        purple_patch
    ]

    colors = ['r', 'g', 'purple']

    if isinstance(movement, SinMovement):
        file_name_movement = list([str(round(float(x))) for x in movement.amplitude_x])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.amplitude_w])
        file_name_movement = 'sin_' + '_'.join(file_name_movement)
    else:
        file_name_movement = list([str(round(float(x))) for x in movement.moving_deltas[0]])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.rotation_deltas[0]])
        file_name_movement = 'linear_' + '_'.join(file_name_movement)

    axis_value = ['x', 'y', 'z']
    res_ = [res[0] for res in resolutions]
    for axis_index in range(0, 3):
        mean = []
        std = []
        heights_val = []
        for index in range(0, int(len(res_) / 100)):
            mean.append(np.mean([delta[axis_index] for delta in deltas][index * 100: (index + 1) * 100]))
            std.append(np.std([delta[axis_index] for delta in deltas][index * 100: (index + 1) * 100]))
            heights_val.append(f'{(index + 1) * 100}')

        mean_file_template = '{0}/w{1}_{2}_{3}_{4}_mean.png'
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot(heights_val, mean, linewidth=3)
        plt.xlabel('Разрешение, пиксели')
        plt.ylabel(f'Среднее значение ошибки угла w{axis_value[axis_index]}, градусы')
        plt.savefig(
            mean_file_template.format(
                plot_config['plot_dir'],
                axis_value[axis_index],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        plt.show()

        std_file_template = '{0}/w{1}_{2}_{3}_{4}_std.png'
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot(heights_val, std, linewidth=3)
        plt.xlabel('Разрешение, пиксели')
        plt.ylabel(f'Значение СКО ошибки w{axis_value[axis_index]}, градусы')
        plt.savefig(
            std_file_template.format(
                plot_config['plot_dir'],
                axis_value[axis_index],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        plt.show()

    for (index, file_name_template, patch, color) in zip(
            list(range(len(file_templates))), file_templates, patches, colors
    ):
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot([res[0] for res in resolutions], [delta[index] for delta in deltas], color, linewidth=3)
        plt.xlabel('Разрешение, пиксели')
        plt.ylabel('Ошибка, градусы')
        plt.legend(handles=[patch], loc='upper right')
        plt.savefig(
            file_name_template.format(
                plot_config['plot_dir'],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        if plot_config['show_plots']:
            plt.show()


def show_plots_position(initial_position, real_positions, calculated_positions, movement, plot_config, simulation_config):
    def get_axe_name(_index):
        if _index == 0:
            return 'X'
        if _index == 1:
            return 'Y'
        if _index == 2:
            return 'Z'

    red_patch = mpatches.Patch(color='red', label='Изменение положения вдоль оси X')
    green_patch = mpatches.Patch(color='green', label='Изменение положения вдоль оси Y')
    purple_patch = mpatches.Patch(color='purple', label='Изменение положения вдоль оси Z')

    plot_size = plot_config['plot_size']
    dpi = plot_config['dpi']
    fig_size = tuple([pix / dpi for pix in plot_size])

    file_templates = [
        '{0}/x_{1}_{2}_{3}.png',
        '{0}/y_{1}_{2}_{3}.png',
        '{0}/z_{1}_{2}_{3}.png'
    ]

    file_templates_delta = [
        '{0}/dx_{1}_{2}_{3}.png',
        '{0}/dy_{1}_{2}_{3}.png',
        '{0}/dz_{1}_{2}_{3}.png'
    ]

    patches = [
        red_patch,
        green_patch,
        purple_patch
    ]

    colors = ['r', 'g', 'purple']

    if isinstance(movement, SinMovement):
        file_name_movement = list([str(round(float(x))) for x in movement.amplitude_x])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.amplitude_w])
        file_name_movement = 'sin_' + '_'.join(file_name_movement)
    else:
        file_name_movement = list([str(round(float(x))) for x in movement.moving_deltas[0]])
        file_name_movement = file_name_movement + list(
            [str(round(float(w) * 180 / math.pi)) for w in movement.rotation_deltas[0]])
        file_name_movement = 'linear_' + '_'.join(file_name_movement)

    real_centered = []
    for (r, i) in zip(real_positions, [initial_position] * len(real_positions)):
        tmp = np.zeros([3, 1])
        delta = r - i
        tmp[0] = delta[0]
        tmp[1] = delta[1]
        tmp[2] = r[2]
        real_centered.append(tmp)

    deltas = [r - c for (r, c) in zip(real_centered, calculated_positions)]

    for (index, patch, color, file_name_template) in zip(range(3), patches, colors, file_templates):
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot([calculated[index] for calculated in calculated_positions], color, linewidth=3)
        plt.plot([real[index] for real in real_centered], 'b', linewidth=3)
        plt.xlabel('Итерация')
        plt.ylabel('Изменение положения вдоль оси {0}'.format(get_axe_name(index)))
        plt.legend(handles=[patch], loc='upper right')
        plt.savefig(
            file_name_template.format(
                plot_config['plot_dir'],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        if plot_config['show_plots']:
            plt.show()

    red_patch = mpatches.Patch(color='red', label='Ошибка определения изменения положения вдоль оси X')
    green_patch = mpatches.Patch(color='green', label='Ошибка определения изменения положения вдоль оси Y')
    purple_patch = mpatches.Patch(color='purple', label='Ошибка определения изменения положения вдоль оси Z')

    patches = [
        red_patch,
        green_patch,
        purple_patch
    ]

    for (index, patch, color, file_name_template) in zip(range(3), patches, colors, file_templates_delta):
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot([delta[index] for delta in deltas], color, linewidth=3)
        plt.xlabel('Итерация')
        plt.ylabel('Отклонения вдоль оси {0}'.format(get_axe_name(index)))
        plt.legend(handles=[patch], loc='upper right')
        plt.savefig(
            file_name_template.format(
                plot_config['plot_dir'],
                simulation_config['type'],
                movement.movement_points,
                file_name_movement
            ),
            bbox_inches='tight'
        )
        if plot_config['show_plots']:
            plt.show()

    print(f'pos: {[calculated[0] for calculated in calculated_positions][-1]}, {[calculated[1] for calculated in calculated_positions][-1]}')
    print(f'pos delta: {[delta[0] for delta in deltas][-1]}, {[delta[1] for delta in deltas][-1]}')
