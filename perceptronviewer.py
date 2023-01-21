import sys
import time
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.qt_compat import QtWidgets
import matplotlib.pyplot as plt
import threading
from perceptron import Perceptron


class PerceptronViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.dataset = gen_linearly_separable()
        self.perceptron = Perceptron([-0.5, -0.5, 1], 0.5, self.dataset, 100, self.update_line)
        self.figure = plt.gcf()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_left_click)
        self.axes = self.figure.add_subplot(111)

        self.decision_boundary_x = np.array([-4, 4])
        self.decision_boundary, = self.axes.plot(self.decision_boundary_x, weights_to_y([-0.5, -0.5, 1], self.decision_boundary_x))

        layout_canvas = QtWidgets.QVBoxLayout(self)
        layout_canvas.addWidget(self.canvas)

        self.plot_dataset()

        t = threading.Thread(target=self.perceptron.run_perceptron)
        t.start()

    def _on_left_click(self, event):
        self.axes.scatter(event.xdata, event.ydata)
        self.figure.canvas.draw()

    def plot_dataset(self):
        markers = ['bo', 'r+']
        for point in self.dataset:
            self.axes.plot(point[0], point[1], markers[int(point[2] - 1)])

    def update_line(self, w):
        self.decision_boundary.set_ydata(weights_to_y(w, self.decision_boundary_x))
        self.figure.canvas.draw()


def weights_to_y(w, x):
    return x * -w[0]/w[1] - w[2]/w[1]


def gen_linearly_separable():
    dataset = np.array([[]])
    min_point = -4
    max_point = 4

    sep_axis = np.random.choice([0, 1], 1)[0]  # 0 - Separated on X axis, 1 - Separated on Y axis
    sep_point = np.random.uniform(min_point, max_point, 1)[0]  # The point at which the two classes are separated

    # Generate minimum/maximum x/y values from randomly chosen axis/separation point
    min_x = [-4, -4 if sep_axis != 0 else sep_point + 0.1]
    max_x = [4 if sep_axis != 0 else sep_point - 0.1, 4]

    min_y = [-4, -4 if sep_axis != 1 else sep_point + 0.1]
    max_y = [4 if sep_axis != 1 else sep_point - 0.1, 4]

    # Randomly generate the points based on the minimums/maximums
    for i in range(2):
        x_vals = np.random.uniform(min_x[i], max_x[i], 50)
        y_vals = np.random.uniform(min_y[i], max_y[i], 50)

        generated_points = np.dstack((x_vals, y_vals, np.full((1, 50), i + 1)))[0]
        dataset = generated_points if dataset.size == 0 else np.concatenate((dataset, generated_points), axis=0)

    return dataset


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    win = PerceptronViewer()
    win.show()
    sys.exit(app.exec_())
