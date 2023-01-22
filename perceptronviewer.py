"""This is the perceptron visualisation module.

It is responsible for all things relating to user interaction (choosing points,
initial weights, learning rate, etc.) and also for showing the output of the
perceptron learning algorithm.
"""

__all__ = ["PerceptronViewer"]
__version__ = "1.0.1.0"
__authors__ = "Kush Bharakhada and Jack Sanders"

import sys
import threading

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from perceptron import Perceptron, PerceptronSettings

CLASS_MARKERS = ["bo", "r+"]

MIN_X = -1000
MAX_X = 1000
MAX_THREADS = 2 # Number of threads allowed to run

X_PLOTS = np.array([MIN_X, MAX_X])

SEPARATOR_COLOUR = "#c0c0c0"

VIS_SPEEDS = [("Slowest", 0.5), ("Slower", 0.25), ("Normal", 0.1),
              ("Faster", 0.05), ("Fastest", 0.01)]


class Separator(QtWidgets.QFrame):
    """A line used to visually separate elements of a GUI"""
    def __init__(self, shape, width):
        super().__init__()
        self.setFrameShape(shape)
        self.setLineWidth(width)
        self.setStyleSheet(f"background-color: {SEPARATOR_COLOUR}; "
                           f"color: {SEPARATOR_COLOUR}")


class PerceptronViewer(QtWidgets.QWidget):
    """PerceptronViewer class used to control the window in which the plot sits.

    This window will contain all user interaction elements (buttons, text boxes,
    sliders, etc.), as well as the actual plot of training data and the decision
    boundary.
    """

    def __init__(self):
        super().__init__()

        self.figure = plt.gcf()
        self.axes = self.figure.add_subplot(111)

        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(-10, 10)

        self.perceptron = None
        self.decision_boundary = None

        self.dataset = []

        layout_canvas = QtWidgets.QVBoxLayout(self)

        options = QtWidgets.QHBoxLayout()

        # Form to control axis limits
        ax_container = QtWidgets.QVBoxLayout()

        ax_lim_form = QtWidgets.QFormLayout()
        ax_lim_form.setLabelAlignment(Qt.AlignRight)

        ax_title = QtWidgets.QLabel("Update Axis Limits")
        ax_title.setStyleSheet("font-size: 18px; text-decoration: underline")

        ax_warning_text = QtWidgets.QLabel(self)
        ax_warning_text.setStyleSheet("color: red; font-size: 14px")

        error_width = ax_warning_text.fontMetrics()\
            .boundingRect(" Please enter a number for each dimension! ").width()
        error_width = int(error_width * 1.25)

        ax_warning_text.setFixedWidth(error_width)

        ax_x_lower = QtWidgets.QLineEdit(self)
        ax_x_lower.setValidator(QIntValidator(-999, 999))

        ax_x_upper = QtWidgets.QLineEdit(self)
        ax_x_upper.setValidator(QIntValidator(-999, 999))

        ax_y_lower = QtWidgets.QLineEdit(self)
        ax_y_lower.setValidator(QIntValidator(-999, 999))

        ax_y_upper = QtWidgets.QLineEdit(self)
        ax_y_upper.setValidator(QIntValidator(-999, 999))

        self.__axis_form = {
            "warning_text": ax_warning_text,
            "x_min": ax_x_lower,
            "x_max": ax_x_upper,
            "y_min": ax_y_lower,
            "y_max": ax_y_upper
        }

        submit_ax_button = QtWidgets.QPushButton("Update Axes")
        submit_ax_button.clicked.connect(self.update_axes)

        ax_lim_form.addRow("Axis X Lower Bound: ", ax_x_lower)
        ax_lim_form.addRow("Axis X Upper Bound: ", ax_x_upper)
        ax_lim_form.addRow("Axis Y Lower Bound: ", ax_y_lower)
        ax_lim_form.addRow("Axis Y Upper Bound: ", ax_y_upper)
        ax_lim_form.addWidget(submit_ax_button)

        ax_container.addWidget(ax_title)
        ax_container.addSpacing(10)
        ax_container.addWidget(ax_warning_text)
        ax_container.addSpacing(5)
        ax_container.addLayout(ax_lim_form)
        ax_container.addSpacing(20)

        # Form to add points to dataset
        point_container = QtWidgets.QVBoxLayout()
        point_add_form = QtWidgets.QFormLayout()
        point_add_form.setLabelAlignment(Qt.AlignRight)

        point_title = QtWidgets.QLabel("Add Points to Dataset")
        point_title.setStyleSheet("font-size: 18px; text-decoration: underline")

        point_warning_text = QtWidgets.QLabel(self)
        point_warning_text.setStyleSheet("color: red; font-size: 14px")
        point_warning_text.setFixedWidth(error_width)

        point_x = QtWidgets.QLineEdit(self)
        point_x.setValidator(QDoubleValidator(-10, 10, 4))

        point_y = QtWidgets.QLineEdit(self)
        point_y.setValidator(QDoubleValidator(-10, 10, 4))

        point_class = QtWidgets.QComboBox(self)
        point_class.addItems(["1", "2"])

        self.__point_form = {
            "warning_text": point_warning_text,
            "x": point_x,
            "y": point_y,
            "class_label": point_class
        }

        submit_point_button = QtWidgets.QPushButton("Add Point")
        submit_point_button.clicked.connect(self.add_point)

        or_box = QtWidgets.QHBoxLayout()
        left_line = Separator(QtWidgets.QFrame.HLine, 2)

        or_label = QtWidgets.QLabel("OR", self)
        or_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        or_label.setStyleSheet("font-size: 16px")

        right_line = Separator(QtWidgets.QFrame.HLine, 2)

        random_dataset_button = QtWidgets.QPushButton(
            "Generate A Random Dataset")
        random_dataset_button.clicked.connect(self.generate_random_dataset)

        or_box.addWidget(left_line)
        or_box.addSpacing(5)
        or_box.addWidget(or_label)
        or_box.addSpacing(5)
        or_box.addWidget(right_line)

        point_add_form.addRow("Point X: ", point_x)
        point_add_form.addRow("Point Y: ", point_y)
        point_add_form.addRow("Point Class: ", point_class)
        point_add_form.addWidget(submit_point_button)

        point_container.addWidget(point_title)
        point_container.addSpacing(10)
        point_container.addWidget(point_warning_text)
        point_container.addSpacing(5)
        point_container.addLayout(point_add_form)
        point_container.addSpacing(10)
        point_container.addLayout(or_box)
        point_container.addSpacing(10)
        point_container.addWidget(random_dataset_button)
        point_container.addSpacing(20)

        # Form for updating perceptron settings and running algorithm
        settings_container = QtWidgets.QVBoxLayout()

        p_settings_form = QtWidgets.QFormLayout()
        p_settings_form.setLabelAlignment(Qt.AlignRight)

        settings_title = QtWidgets.QLabel("Run Perceptron")
        settings_title.setStyleSheet(
            "font-size: 18px; text-decoration: underline")

        settings_warning_text = QtWidgets.QLabel(self)
        settings_warning_text.setStyleSheet("color: red; font-size: 14px")
        settings_warning_text.setFixedWidth(error_width)
        settings_warning_text.setWordWrap(True)

        w_1_line_field = QtWidgets.QLineEdit(self)
        w_1_line_field.setValidator(QDoubleValidator(-999, 999, 4))

        w_2_line_field = QtWidgets.QLineEdit(self)
        w_2_line_field.setValidator(QDoubleValidator(-999, 999, 4))

        w_0_line_field = QtWidgets.QLineEdit(self)
        w_0_line_field.setValidator(QDoubleValidator(-999, 999, 4))

        learning_line_field = QtWidgets.QLineEdit(self)
        learning_line_field.setValidator(QDoubleValidator(0, 10, 2))

        iteration_limit_field = QtWidgets.QLineEdit(self)
        iteration_limit_field.setValidator(QIntValidator(1, 1000000))

        speed_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        speed_slider.setMinimum(0)
        speed_slider.setMaximum(len(VIS_SPEEDS) - 1)
        speed_slider.setSingleStep(1)
        speed_slider.setValue(len(VIS_SPEEDS) // 2)

        def slider_update(i):
            update_form_label(p_settings_form, speed_slider,
                              f"Visualisation Speed: {VIS_SPEEDS[i][0]}")

        speed_slider.valueChanged.connect(slider_update)

        self.__settings_form = {
            "warning_text": settings_warning_text,
            "w_1": w_1_line_field,
            "w_2": w_2_line_field,
            "w_0": w_0_line_field,
            "learning_rate": learning_line_field,
            "visualisation_speed": speed_slider,
            "iteration_limit": iteration_limit_field,
        }

        run_perceptron_button = QtWidgets.QPushButton("Run Perceptron")
        run_perceptron_button.clicked.connect(self.run_perceptron)

        p_settings_form.addRow("Initial w_1: ", w_1_line_field)
        p_settings_form.addRow("Initial w_2: ", w_2_line_field)
        p_settings_form.addRow("Initial w_0: ", w_0_line_field)
        p_settings_form.addRow("Learning Rate: ", learning_line_field)
        p_settings_form.addRow("Iteration Limit: ", iteration_limit_field)
        p_settings_form.addRow("Visualisation Speed: "
                               + VIS_SPEEDS[len(VIS_SPEEDS) // 2][0],
                               speed_slider)
        p_settings_form.addWidget(run_perceptron_button)

        settings_container.addWidget(settings_title)
        settings_container.addSpacing(10)
        settings_container.addWidget(settings_warning_text)
        settings_container.addSpacing(5)
        settings_container.addLayout(p_settings_form)
        settings_container.addSpacing(20)

        speed_label = p_settings_form.labelForField(speed_slider)
        longest = "Visualisation Speed: " + max(s[0] for s in VIS_SPEEDS)
        long_length = speed_label.fontMetrics().boundingRect(longest).width()
        speed_label.setFixedWidth(long_length)

        ax_container.setAlignment(Qt.AlignTop)
        point_container.setAlignment(Qt.AlignTop)
        settings_container.setAlignment(Qt.AlignTop)

        options.addLayout(ax_container)
        options.addSpacing(14)
        options.addWidget(Separator(QtWidgets.QFrame.VLine, 4))
        options.addSpacing(14)
        options.addLayout(point_container)
        options.addSpacing(14)
        options.addWidget(Separator(QtWidgets.QFrame.VLine, 4))
        options.addSpacing(14)
        options.addLayout(settings_container)

        diagnostic_box = QtWidgets.QVBoxLayout()
        graphAndDiagnosticBox = QtWidgets.QHBoxLayout()

        graphAndDiagnosticBox.addLayout(diagnostic_box)
        graphAndDiagnosticBox.addWidget(FigureCanvas(self.figure))

        for label in DIAGNOSTIC_LABELS.values():
            diagnostic_box.addWidget(label)

        layout_canvas.addLayout(graphAndDiagnosticBox)
        layout_canvas.addSpacing(30)
        layout_canvas.addLayout(options)


    def add_point(self):
        """Takes user input and adds a point at the desired location"""
        warning_text = self.__point_form["warning_text"]
        try:
            x = float(self.__point_form["x"].text())
            y = float(self.__point_form["y"].text())
        except ValueError:
            warning_text.setText("Please enter a number for each dimension!")
            return

        warning_text.setText("")

        class_ = int(self.__point_form["class_label"].currentText())

        self.dataset.append([x, y, class_])
        self.axes.plot(x, y, CLASS_MARKERS[int(class_ - 1)])
        self.figure.canvas.draw()

    def update_axes(self):
        """Takes user input and updates the axis xlim and ylim"""
        warning_text = self.__axis_form["warning_text"]

        try:
            min_x = int(self.__axis_form["x_min"].text())
            max_x = int(self.__axis_form["x_max"].text())
            min_y = int(self.__axis_form["y_min"].text())
            max_y = int(self.__axis_form["y_max"].text())
        except ValueError:
            warning_text.setText("Please enter a number for each bound!")
            return

        if min_x >= max_x:
            warning_text.setText("Minimum X must be <= Maximum X!")
            return
        if min_y >= max_y:
            warning_text.setText("Minimum Y must be <= Maximum Y!")
            return

        warning_text.setText("")

        self.axes.set_xlim(min_x, max_x)
        self.axes.set_ylim(min_y, max_y)

        self.figure.canvas.draw()

    def run_perceptron(self):
        """Instantiates and runs a perceptron using user-inputted settings."""
        warning_text = self.__settings_form["warning_text"]

        # Prevent user from spamming run perceptron
        if threading.active_count() >= MAX_THREADS:
            warning_text.setText("Please wait for this algorithm to end!")
            return

        try:
            w_1 = float(self.__settings_form["w_1"].text())
            w_2 = float(self.__settings_form["w_2"].text())
            w_0 = float(self.__settings_form["w_0"].text())
            learning_rate = float(self.__settings_form["learning_rate"].text())
            iter_limit = int(self.__settings_form["iteration_limit"].text())
            speed = int(self.__settings_form["visualisation_speed"].value())
            vis_speed = VIS_SPEEDS[speed][1]
        except ValueError:
            warning_text.setText("Please enter a value for each setting!")
            return

        counts = [0, 0]
        for point in self.dataset:
            counts[int(point[-1]) - 1] += 1

        if counts.count(0) > 0:
            warning_text.setText("Please ensure your database contains at least"
                                 " one point in each class!")
            return

        warning_text.setText("")

        self.dataset = np.array(self.dataset)

        y = weights_to_y([w_1, w_2, w_0])
        (self.decision_boundary,) = self.axes.plot(X_PLOTS, y)

        p_settings = PerceptronSettings((np.array([w_1, w_2, w_0])),
                                        learning_rate, iter_limit, vis_speed)
        self.perceptron = Perceptron(p_settings, self.dataset, self.update_line)

        self.start_learning()

    def generate_random_dataset(self):
        """Takes user input to generate a random dataset and plot it"""
        self.dataset = gen_linearly_separable(*self.axes.get_xlim(),
                                              *self.axes.get_ylim())
        self.plot_dataset()

    def plot_dataset(self):
        """Plots the dataset used by the perceptron."""
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()

        self.axes.clear()

        self.axes.set_xlim(xlim)
        self.axes.set_ylim(ylim)
        for point in self.dataset:
            self.axes.plot(point[0], point[1], CLASS_MARKERS[int(point[2] - 1)])
        self.figure.canvas.draw()

    def update_line(self, weights, iteration):
        """Update the drawn line based on the weights of the perceptron. Passed
        to the perceptron as the gui_callback attribute.
        :param weights: The weights of the perceptron's decision boundary
        (current iteration)
        """
        DIAGNOSTIC_LABELS["iteration_label"].setText(f"Current Iteration: {iteration}")
        DIAGNOSTIC_LABELS["current_weights"].setText(f"Current Weights: [w1 = {round(weights[0], 4)}] "
                                                     f"[w2 = {round(weights[1], 4)}] [w0 = {round(weights[2], 4)}]")

        new_ydata = weights_to_y(weights)
        self.decision_boundary.set_ydata(new_ydata)
        self.figure.canvas.draw()

    def start_learning(self):
        """Starts a thread for the perceptron learning algorithm."""
        threading.Thread(target=self.perceptron.run_perceptron).start()


def update_form_label(form, for_, text):
    """A small function to update a label within a PyQt5 FormLayout.

    :param form: The form containing the label you are trying to update.
    :param for_: The widget being labelled by the label you want to update.
    :param text: The new text of the label
    """
    form.labelForField(for_).setText(text)


def weights_to_y(weights):
    """Converts perceptron weights (a vector of w_1, w_2, w_0) to y values,
    usable for plotting in matplotlib.
    :param weights: The weights of the current iteration
    :return: The calculated y values
    """
    return X_PLOTS * -weights[0] / weights[1] - weights[2] / weights[1]


def gen_linearly_separable(min_x, max_x, min_y, max_y):
    """Generates a random, linearly separable dataset. Works by first randomly
    choosing the axis in which the dataset should be separable (i.e. if it
    should be separable by a horizontal or vertical line) and then choosing
    the point along that axis at which to split the dataset.
    :return: A dataset of points in the form [x, y, class]
    """
    point_lims = [[min_x, max_x], [min_y, max_y]]

    dataset = np.array([[]])

    # 0 - Separated on X axis, 1 - Separated on Y axis
    sep_axis = np.random.choice([0, 1], 1)[0]

    axis_lims = point_lims[sep_axis]

    # The point at which the two classes are separated
    sep_point = np.random.uniform(axis_lims[0], axis_lims[1], 1)[0]

    # Calculate minimum/maximum x/y values from randomly chosen axis &
    # separation point
    min_x_points = [min_x, min_x if sep_axis != 0 else sep_point + 0.1]
    max_x_points = [max_x if sep_axis != 0 else sep_point - 0.1, max_x]

    min_y_points = [min_y, min_y if sep_axis != 1 else sep_point + 0.1]
    max_y_points = [max_y if sep_axis != 1 else sep_point - 0.1, max_y]

    # Randomly generate the points based on the minimums/maximums
    for i in range(2):
        x_vals = np.random.uniform(min_x_points[i], max_x_points[i], 50)
        y_vals = np.random.uniform(min_y_points[i], max_y_points[i], 50)

        generated_points = np.dstack((x_vals, y_vals, np.full((1, 50), i + 1)))[
            0]
        dataset = generated_points if dataset.size == 0 else np.concatenate(
            (dataset, generated_points), axis=0)

    return dataset


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    DIAGNOSTIC_LABELS = {
        "iteration_label": QtWidgets.QLabel("Current Iteration: 0"),
        "current_weights": QtWidgets.QLabel("Current Weights: [w1 = 1] [w2 = 1] [w0 = 1]"),
    }

    win = PerceptronViewer()
    win.show()
    sys.exit(app.exec_())
