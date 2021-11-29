import matplotlib.pyplot as plt
plt.ion()


class DynamicPlotUpdate:
    def __init__(self, subplots):
        self.figure, self.axes = plt.subplots(subplots)
        self.lines = []
        if subplots > 1:
            for axe in self.axes:
                self.lines.append(axe.plot([], [], 'o')[0])
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


import time
d = DynamicPlotUpdate(subplots=2)
# d.on_launch()
# d.update_data([0, 1, 2], [2, 3, 4])
# time.sleep(3)
# d.update_data([1, 2, 3], [5, 6, 7])
# time.sleep(3)
d()