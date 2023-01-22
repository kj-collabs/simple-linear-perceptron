"""
perceptron.py
Runs the Perceptron algorithm.
"""

__all__ = ["Perceptron", "PerceptronSettings"]
__version__ = "1.2.0.0"
__authors__ = "Kush Bharakhada and Jack Sanders"

import time
import typing

import numpy as np

CLASS_1 = 1
CLASS_2 = -1


class PerceptronSettings(typing.NamedTuple):
    """A named tuple to pass settings to the perceptron"""
    weights: np.ndarray
    learning_rate: float
    iter_limit: int
    visualisation_speed: float


class Perceptron:
    """The Perceptron class is responsible for actually running the perceptron
    learning algorithm.
    """

    def __init__(self, settings: PerceptronSettings, train_data: np.ndarray,
                 gui_update_callback=print):
        """
        Constructor for the Perceptron Algorithm.
        :param settings: An instance of PerceptronSettings
        :param train_data: Training points as [ [x1, x2, class (1 or 2)] ]
        :param gui_update_callback: Used by the GUI
        """
        self.__weights = settings.weights
        self.__learning_rate = settings.learning_rate
        self.__iter_limit = settings.iter_limit
        self.__delay = settings.visualisation_speed
        self.__train_points = self.format_training_points(train_data)
        self.__callback = gui_update_callback

    def set_learning_rate(self, learning):
        """
        Setter for learning rate
        :param learning: Learning Rate
        """
        self.__learning_rate = learning

    def set_train_data(self, train_data):
        """
        Setter for the training data
        :param train_data:
        """
        self.__train_points = train_data

    def set_weights(self, weights):
        """
        Setter for weights
        :param weights: Weights as [w1, w2, w0]
        """
        self.__weights = weights

    def run_perceptron(self):
        """Runs the perceptron learning algorithm, and calls the GUI callback
        function with the weights to update the plotted decision boundary.
        """
        class1_points = self.__train_points[0]
        class2_points = self.__train_points[1]
        finished = False
        iteration = 0

        while not finished and iteration <= self.__iter_limit:
            class1_misclassifications = np.array(list(filter(
                lambda point: CLASS_2 * (self.__weights @ point) >= 0,
                class1_points))) * CLASS_2

            class2_misclassifications = np.array(list(filter(
                lambda point: CLASS_1 * (self.__weights @ point) >= 0,
                class2_points))) * CLASS_1

            misclassified_points = class1_misclassifications.tolist() \
                                   + class2_misclassifications.tolist()

            # Check if misclassifications occurred this iteration
            if len(misclassified_points) == 0:
                finished = True

            # Update weights
            self.__weights -= self.__learning_rate \
                              * np.sum(np.array(misclassified_points), axis=0)
            # Update graph
            self.__callback(self.__weights)
            iteration += 1
            time.sleep(self.__delay)

    @staticmethod
    def format_training_points(training_points):
        # Split the training points into class 1 and class 2
        class1 = training_points[training_points[:, -1] == 1]
        class2 = training_points[training_points[:, -1] == 2]
        # Replace the last item in the array with 1 for w0 rather
        # than the class value 1 or 2.
        class1[:, -1] = 1
        class2[:, -1] = 1
        return class1, class2


# Remove later
if __name__ == "__main__":
    init_weights = np.array([1.0, -1.0, 1.0])
    LEARNING_RATE = 0.5
    train = np.array(
        [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 2]])

    perceptron_settings = PerceptronSettings(init_weights, LEARNING_RATE, 100,
                                             0.25)
    perceptron = Perceptron(perceptron_settings, train)
    perceptron.run_perceptron()
