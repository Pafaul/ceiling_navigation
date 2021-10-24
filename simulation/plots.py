import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def show_plots(real_angles, angles):
    blue_patch = mpatches.Patch(color='blue', label='Действительный угол')
    red_patch = mpatches.Patch(color='red', label='Рассчитанный угол wx')
    green_patch = mpatches.Patch(color='green', label='Рассчитанный угол wy')
    purple_patch = mpatches.Patch(color='purple', label='Рассчитанный угол wz')

    plt.grid(True)
    plt.plot([angle[0] for angle in real_angles], 'b')
    plt.plot([-angle[0] for angle in angles], 'r')
    plt.xlabel('Итерация')
    plt.ylabel('Угол, градусы')
    plt.legend(handles=[blue_patch, red_patch], loc='upper right')
    plt.show()

    plt.grid(True)
    plt.plot([angle[1] for angle in real_angles], 'b')
    plt.plot([-angle[1] for angle in angles], 'g')
    plt.xlabel('Итерация')
    plt.ylabel('Угол, градусы')
    plt.legend(handles=[blue_patch, green_patch], loc='upper right')
    plt.show()

    plt.grid(True)
    plt.plot([angle[2] for angle in real_angles], 'b')
    plt.plot([angle[2] for angle in angles], 'purple')
    plt.xlabel('Итерация')
    plt.ylabel('Угол, градусы')
    plt.legend(handles=[blue_patch, purple_patch], loc='upper right')
    plt.show()

    red_patch = mpatches.Patch(color='red', label='Ошибка wx')
    green_patch = mpatches.Patch(color='green', label='Ошибка wy')
    purple_patch = mpatches.Patch(color='purple', label='Ошибка wz')

    plt.grid(True)
    plt.plot([real_angle[0] + angle[0] for (real_angle, angle) in zip(real_angles, angles)], 'r')
    plt.plot([real_angle[1] + angle[1] for (real_angle, angle) in zip(real_angles, angles)], 'g')
    plt.plot([real_angle[2] - angle[2] for (real_angle, angle) in zip(real_angles, angles)], 'purple')
    plt.xlabel('Итерация')
    plt.ylabel('Ошибка определения угла, градусы')
    plt.legend(handles=[red_patch, green_patch, purple_patch], loc='upper right')
    plt.show()


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

    file_name_movement = list([str(round(float(w) * 180 / math.pi)) for w in movement.amplitude_w])
    file_name_movement = '_'.join(file_name_movement)

    for (index, file_name_template, patch, color) in zip(
            list(range(len(file_templates))), file_templates, patches, colors
    ):
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot(heights, [delta[index] for delta in deltas], color)
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

    file_name_movement = list([str(round(float(w) * 180 / math.pi)) for w in movement.amplitude_w])
    file_name_movement = '_'.join(file_name_movement)

    for (index, file_name_template, patch, color) in zip(
            list(range(len(file_templates))), file_templates, patches, colors
    ):
        plt.figure(figsize=fig_size, dpi=dpi)
        plt.grid(True)
        plt.plot([res[0] for res in resolutions], [delta[index] for delta in deltas], color)
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
