import numpy as np
import time
"""
perceptron.py
Runs the Perceptron algorithm.
"""


class Perceptron(object):
    def __init__(self, w, p, train, limit, gui_update_callback):
        """
        Constructor for the Perceptron Algorithm.
        :param w: Weights in order [w1, w2, w0 (bias)]
        :param p: Learning Rate
        :param train: Training points as [ [x1, x2, class (1 or 2)] ]
        :param limit: Iteration limit
        :param gui_update_callback: Used by the GUI
        """
        self.__weights = w
        self.__learning_rate = p
        self.__train_points = self.__format_training_points(train)
        self.__iter_limit = limit
        self.__callback = gui_update_callback

    def set_learning_rate(self, p):
        """
        Setter for learning rate
        :param p: Learning Rate
        """
        self.__learning_rate = p

    def set_train_data(self, train_data):
        """
        Setter for the training data
        :param train_data:
        """
        self.__train_points = train_data

    def set_weights(self, w):
        """
        Setter for weights
        :param w: Weights as [w1, w2, w0]
        """
        self.__weights = w

    def run_perceptron(self):
        class1_points, class2_points = self.__train_points[0], self.__train_points[1]
        CLASS1, CLASS2 = 1, -1
        finished = False
        iteration = 0

        while not finished and iteration <= self.__iter_limit:
            class1_misclassifications = np.array(list(filter(
                lambda point: CLASS2 * (self.__weights @ point) >= 0, class1_points))) * CLASS2

            class2_misclassifications = np.array(list(filter(
                lambda point: CLASS1 * (self.__weights @ point) >= 0, class2_points))) * CLASS1

            misclassified_points = class1_misclassifications.tolist() + class2_misclassifications.tolist()

            # Check if misclassifications occurred this iteration
            if len(misclassified_points) == 0:
                finished = True

            # Update weights
            self.__weights = self.__weights - self.__learning_rate * np.sum(np.array(misclassified_points), axis=0)
            # Update graph
            self.__callback(self.__weights)
            iteration += 1
            time.sleep(0.1)

    def __format_training_points(self, training_points):
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
    w = np.array([1.0, -1.0, 0])
    p = 0.1
    train = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 2]])

    perceptron = Perceptron(w, p, train, 100, lambda x: print(x))
    perceptron.run_perceptron()
