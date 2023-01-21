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

        self.perceptron = Perceptron(lambda xdata, ydata: self.update_line(xdata, ydata))
        self.figure = plt.gcf()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_left_click)
        self.axes = self.figure.add_subplot(111)

        x = np.arange(0, 10, 0.1)
        y = np.cos(x)
        self.decision_boundary, = self.axes.plot(x, y)

        layout_canvas = QtWidgets.QVBoxLayout(self)
        layout_canvas.addWidget(self.canvas)

        t = threading.Thread(target=self.perceptron.run_perceptron)
        t.start()

    def _on_left_click(self, event):
        self.axes.scatter(event.xdata, event.ydata)
        self.figure.canvas.draw()

    def update_line(self, x, y):
        self.decision_boundary.set_xdata(x)
        self.decision_boundary.set_ydata(y)
        self.figure.canvas.draw()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = PerceptronViewer()
    w.show()
    sys.exit(app.exec_())
