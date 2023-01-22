"""This is the perceptron visualisation module.

It is responsible for all things relating to user interaction (choosing points,
initial weights, learning rate, etc.) and also for showing the output of the
perceptron learning algorithm.
"""

__all__ = ["PerceptronViewer"]
__version__ = "1.2.0.0"
__authors__ = "Kush Bharakhada and Jack Sanders"

import sys
import threading

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont, QFontMetrics

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


class PyQtForm(QtWidgets.QVBoxLayout):
    """A generic form class to keep the code in PerceptronViewer clean."""

    def __init__(self, title, fields, validators, btn_text, on_submit_action):
        super().__init__()

        # Break fields parameter up to titles, ids, and custom widgets (if any)
        field_titles = [a for (a, _, _) in fields]
        field_ids = [b for (_, b, _) in fields]
        customs = [c for (_, _, c) in fields]

        form_title = QtWidgets.QLabel(title)
        form_title.setFont(TITLE_FONT)

        self.warning_text = QtWidgets.QLabel()
        self.warning_text.setMinimumWidth(MIN_FORM_WIDTH)
        self.warning_text.setFont(ERROR_FONT)
        self.warning_text.setStyleSheet("color: red")
        self.warning_text.setWordWrap(True)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        # Store fields and corrosponding labels in a dict for easy access
        self.fields = {}
        self.labels = {}
        for i in range(len(fields)):
            if customs[i] is not None:
                c_field = customs[i]  # Get custom widget (if one was passed)
                form.addRow(field_titles[i] + ": ", c_field)
                self.fields[field_ids[i]] = c_field
                self.labels[field_ids[i]] = form.labelForField(c_field)

                c_field.setFont(LABEL_FONT)
                form.labelForField(c_field).setFont(LABEL_FONT)
            else:
                current_field = QtWidgets.QLineEdit()
                current_field.setValidator(validators[i])
                form.addRow(field_titles[i] + ": ", current_field)
                self.fields[field_ids[i]] = current_field
                self.labels[field_ids[i]] = form.labelForField(current_field)

                current_field.setFont(LABEL_FONT)
                form.labelForField(current_field).setFont(LABEL_FONT)

        submit_button = QtWidgets.QPushButton(btn_text)
        submit_button.clicked.connect(on_submit_action)
        form.addWidget(submit_button)

        self.addWidget(form_title)
        self.addSpacing(5)
        self.addWidget(self.warning_text)
        self.addLayout(form)


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
        self.__ax_form = PyQtForm("Update Axis Limits",
                                  [("Axis X Lower Bound", "x_min", None),
                                   ("Axis X Upper Bound", "x_max", None),
                                   ("Axis Y Lower Bound", "y_min", None),
                                   ("Axis Y Upper Bound", "y_max", None)],
                                  [QIntValidator(-999, 999) for _ in range(4)],
                                  "Update Axes", self.update_axes)

        # Form to add points to dataset
        point_class = QtWidgets.QComboBox(self)
        point_class.addItems(["1", "2"])

        self.__point_form = PyQtForm("Add Points to Dataset",
                                     [("Point X", "x", None),
                                      ("Point Y", "y", None),
                                      ("Point Class", "class", point_class)],
                                     [QDoubleValidator(-10, 10, 4),
                                      QDoubleValidator(-10, 10, 4)],
                                     "Add Point", self.add_point)

        point_container = QtWidgets.QVBoxLayout()

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

        point_container.addLayout(self.__point_form)
        point_container.addSpacing(10)
        point_container.addLayout(or_box)
        point_container.addSpacing(10)
        point_container.addWidget(random_dataset_button)
        point_container.addSpacing(20)

        # Form for updating perceptron settings and running algorithm
        # Create QSlider to control visualisation speed
        speed_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        speed_slider.setMinimum(0)
        speed_slider.setMaximum(len(VIS_SPEEDS) - 1)
        speed_slider.setSingleStep(1)
        speed_slider.setValue(len(VIS_SPEEDS) // 2)

        # When speed changed, update the label to reflect the current selection.
        def slider_update(i):
            speed = VIS_SPEEDS[i][0]
            speed_text = "Visualisation Speed: " + speed
            self.__settings_form.labels["speed"].setText(speed_text)

        speed_slider.valueChanged.connect(slider_update)

        init_speed = "Visualisation Speed: "\
                     + VIS_SPEEDS[len(VIS_SPEEDS) // 2][0]

        self.__settings_form = PyQtForm("Run Perceptron",
                                        [("w_1", "w_1", None),
                                         ("w_2", "w_2", None),
                                         ("w_0", "w_0", None),
                                         ("Learning Rate", "learning", None),
                                         ("Iteration Limit", "iter_lim", None),
                                         (init_speed, "speed", speed_slider)],
                                        [QDoubleValidator(-999, 999, 4),
                                         QDoubleValidator(-999, 999, 4),
                                         QDoubleValidator(-999, 999, 4),
                                         QDoubleValidator(0, 10, 2),
                                         QIntValidator(1, 1000000), None],
                                        "Run Perceptron", self.run_perceptron)

        # Fix width of speed label, to prevent form expanding/shrinking when
        # speed value changes.
        speed_label = self.__settings_form.labels["speed"]
        longest = "Visualisation Speed: " + max(s[0] for s in VIS_SPEEDS)
        long_length = speed_label.fontMetrics().boundingRect(longest).width()
        speed_label.setFixedWidth(long_length)

        self.__ax_form.setAlignment(Qt.AlignTop)
        point_container.setAlignment(Qt.AlignTop)
        self.__settings_form.setAlignment(Qt.AlignTop)

        options.addLayout(self.__ax_form)
        options.addSpacing(14)
        options.addWidget(Separator(QtWidgets.QFrame.VLine, 4))
        options.addSpacing(14)
        options.addLayout(point_container)
        options.addSpacing(14)
        options.addWidget(Separator(QtWidgets.QFrame.VLine, 4))
        options.addSpacing(14)
        options.addLayout(self.__settings_form)

        diagnostic_label = QtWidgets.QLabel("Diagnostics")
        diagnostic_label.setStyleSheet("font-size: 18px; text-decoration: underline")
        diagnostic_label.setAlignment(Qt.AlignCenter)
        diagnostic_box = QtWidgets.QVBoxLayout()
        diagnostic_box.setAlignment(Qt.AlignVCenter)
        diagnostic_box.addWidget(diagnostic_label)
        graphAndDiagnosticBox = QtWidgets.QHBoxLayout()
        graphAndDiagnosticBox.addLayout(diagnostic_box)
        graphAndDiagnosticBox.addWidget(FigureCanvas(self.figure))

        for label in DIAGNOSTIC_LABELS.values():
            label.setMaximumWidth(200)
            label.setAlignment(Qt.AlignCenter)
            diagnostic_box.addWidget(label)

        layout_canvas.addLayout(graphAndDiagnosticBox)
        layout_canvas.addSpacing(30)
        layout_canvas.addLayout(options)


    def add_point(self):
        """Takes user input and adds a point at the desired location"""
        warning_text = self.__point_form.warning_text
        try:
            x = float(self.__point_form.fields["x"].text())
            y = float(self.__point_form.fields["y"].text())
        except ValueError:
            warning_text.setText("Please enter a number for each dimension!")
            return

        warning_text.setText("")

        class_ = int(self.__point_form.fields["class"].currentText())

        self.dataset.append([x, y, class_])
        self.axes.plot(x, y, CLASS_MARKERS[int(class_ - 1)])
        self.figure.canvas.draw()

    def update_axes(self):
        """Takes user input and updates the axis xlim and ylim"""
        warning_text = self.__ax_form.warning_text

        try:
            min_x = int(self.__ax_form.fields["x_min"].text())
            max_x = int(self.__ax_form.fields["x_max"].text())
            min_y = int(self.__ax_form.fields["y_min"].text())
            max_y = int(self.__ax_form.fields["y_max"].text())
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
        warning_text = self.__settings_form.warning_text

        # Prevent user from spamming run perceptron
        if threading.active_count() >= MAX_THREADS:
            warning_text.setText("Please wait for this algorithm to end!")
            return

        try:
            w_1 = float(self.__settings_form.fields["w_1"].text())
            w_2 = float(self.__settings_form.fields["w_2"].text())
            w_0 = float(self.__settings_form.fields["w_0"].text())

            learning_rate = self.__settings_form.fields["learning"].text()
            learning_rate = float(learning_rate)

            iter_limit = self.__settings_form.fields["iter_lim"].text()
            iter_limit = int(iter_limit)

            speed = self.__settings_form.fields["speed"].value()
            vis_speed = VIS_SPEEDS[speed][1]

        except ValueError:
            warning_text.setText("Please enter a value for each setting!")
            return

        counts = [0, 0]
        for point in self.dataset:
            counts[int(point[-1]) - 1] += 1

        if counts.count(0) > 0:
            warning_text.setText("Please ensure your dataset contains at least"
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
        DIAGNOSTIC_LABELS["current_weights"].setText(f"Current Weights: \n[w1 = {round(weights[0], 4)}] "
                                                     f"\n[w2 = {round(weights[1], 4)}] \n[w0 = {round(weights[2], 4)}]")

        new_ydata = weights_to_y(weights)
        self.decision_boundary.set_ydata(new_ydata)
        self.figure.canvas.draw()

    def start_learning(self):
        """Starts a thread for the perceptron learning algorithm."""
        threading.Thread(target=self.perceptron.run_perceptron).start()


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

    TITLE_FONT = QFont('Arial', 16)
    TITLE_FONT.setUnderline(True)
    TITLE_FONT.setCapitalization(QFont.Capitalization.Capitalize)
    ERROR_FONT = QFont('Arial', 12)
    LABEL_FONT = QFont('Arial', 10)
    MIN_FORM_WIDTH = QFontMetrics(ERROR_FONT) \
        .boundingRect(" Please enter a number for each dimension! ").width()

    DIAGNOSTIC_LABELS = {
        "iteration_label": QtWidgets.QLabel("Current Iteration: 0"),
        "current_weights": QtWidgets.QLabel("Current Weights: \n[w1 = 0] \n[w2 = 0] \n[w0 = 0]"),
    }

    win = PerceptronViewer()

    win.show()
    sys.exit(app.exec_())
