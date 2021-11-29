from matplotlib import pyplot as plt


class DynamicPlotUpdate:
    def __init__(self, subplots, marker_type):
        plot_size = [1600, 900]
        dpi = 100
        fig_size = tuple([pix / dpi for pix in plot_size])
        self.figure, self.axes = plt.subplots(subplots, figsize=fig_size, dpi=dpi)
        self.lines = []
        if subplots > 1:
            for axe in self.axes:
                self.lines.append(axe.plot([], [], marker_type)[0])
                axe.set_autoscalex_on(True)
                axe.set_autoscaley_on(True)
                axe.grid()

    def on_running(self, plot_data):
        for (line, ax, data) in zip(self.lines, self.axes, plot_data):
            line.set_xdata(data[0])
            line.set_ydata(data[1])
            ax.relim()
            ax.autoscale_view()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def update_data(self, plot_data):
        self.on_running(plot_data)