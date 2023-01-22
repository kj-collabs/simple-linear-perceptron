"""This is the perceptron visualisation module.

It is responsible for all things relating to user interaction (choosing points,
initial weights, learning rate, etc.) and also for showing the output of the
perceptron learning algorithm.
"""

__all__ = ["PerceptronViewer"]
__version__ = "1.0.0.0"
__authors__ = "Kush Bharakhada and Jack Sanders"

import sys
import threading

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets

from perceptron import Perceptron


class PerceptronViewer(QtWidgets.QWidget):
    """PerceptronViewer class used to control the window in which the plot sits.

    This window will contain all user interaction elements (buttons, text boxes,
    sliders, etc.), as well as the actual plot of training data and the decision
    boundary.
    """
    def __init__(self):
        super().__init__()

        self.dataset = gen_linearly_separable()
        self.perceptron = Perceptron([-0.5, -0.5, 1], 0.5, self.dataset, 100,
                                     self.update_line)
        self.figure = plt.gcf()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_left_click)
        self.axes = self.figure.add_subplot(111)

        self.decision_boundary_x = np.array([-4, 4])
        y = weights_to_y([-0.5, -0.5, 1], self.decision_boundary_x)
        (self.decision_boundary,) = self.axes.plot(self.decision_boundary_x, y)

        layout_canvas = QtWidgets.QVBoxLayout(self)
        layout_canvas.addWidget(self.canvas)

        self.plot_dataset()

        threading.Thread(target=self.perceptron.run_perceptron).start()

    def _on_left_click(self, event):
        self.axes.scatter(event.xdata, event.ydata)
        self.figure.canvas.draw()

    def plot_dataset(self):
        """Plots the dataset used by the perceptron."""
        markers = ["bo", "r+"]
        for point in self.dataset:
            self.axes.plot(point[0], point[1], markers[int(point[2] - 1)])

    def update_line(self, weights):
        """Update the drawn line based on the weights of the perceptron. Passed
        to the perceptron as the gui_callback attribute.
        :param weights: The weights of the perceptron's decision boundary
        (current iteration)
        """
        new_ydata = weights_to_y(weights, self.decision_boundary_x)
        self.decision_boundary.set_ydata(new_ydata)
        self.figure.canvas.draw()


def weights_to_y(weights, x):
    """Converts perceptron weights (a vector of w1, w2, w0) to y values, usable
    for plotting in matplotlib.
    :param weights: The weights of the current iteration
    :param x: The x values for which to calculate the corrosponding y values
    :return: The calculated y values
    """
    return x * -weights[0] / weights[1] - weights[2] / weights[1]


def gen_linearly_separable():
    """Generates a random, linearly separable dataset. Works by first randomly
    choosing the axis in which the dataset should be separable (i.e. if it
    should be separable by a horizontal or vertical line) and then choosing
    the point along that axis at which to split the dataset.
    :return: A dataset of points in the form [x, y, class]
    """
    dataset = np.array([[]])
    min_point = -4
    max_point = 4

    # 0 - Separated on X axis, 1 - Separated on Y axis
    sep_axis = np.random.choice([0, 1], 1)[0]

    # The point at which the two classes are separated
    sep_point = np.random.uniform(min_point, max_point, 1)[0]

    # Calculate minimum/maximum x/y values from randomly chosen axis &
    # separation point
    min_x = [-4, -4 if sep_axis != 0 else sep_point + 0.1]
    max_x = [4 if sep_axis != 0 else sep_point - 0.1, 4]

    min_y = [-4, -4 if sep_axis != 1 else sep_point + 0.1]
    max_y = [4 if sep_axis != 1 else sep_point - 0.1, 4]

    # Randomly generate the points based on the minimums/maximums
    for i in range(2):
        x_vals = np.random.uniform(min_x[i], max_x[i], 50)
        y_vals = np.random.uniform(min_y[i], max_y[i], 50)

        generated_points = np.dstack((x_vals, y_vals, np.full((1, 50), i + 1)))[
            0]
        dataset = generated_points if dataset.size == 0 else np.concatenate(
            (dataset, generated_points), axis=0)

    return dataset


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    win = PerceptronViewer()
    win.show()
    sys.exit(app.exec_())
