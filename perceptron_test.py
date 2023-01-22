"""
perceptron_test.py
Some tests for the perceptron.py class.
"""

__all__ = ["Perceptron", "PerceptronSettings"]
__version__ = "1.0.1.0"
__authors__ = "Kush Bharakhada and Jack Sanders"

import unittest

import numpy as np
from perceptron import Perceptron, PerceptronSettings


class TestPerceptron(unittest.TestCase):
    """ A basic test suite for the perceptron learning algorithm."""
    def setUp(self):
        """setUp function called by the __init__ function of TestCase to set
        instance variables for the current test suite.
        """
        self.weights = np.array([1.0, -1.0, 0])
        self.learning_rate = 0.1
        self.train_points = np.array(
            [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 2, 2], [2, 1, 2], [2, 2, 2]])
        self.formatted_training_points = np.array(
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]]), \
            np.array([[1, 2, 1], [2, 1, 1], [2, 2, 1]])
        self.iterations = 100
        self.final_weights = np.array([-0.2, -0.6, 0.7])

    def create_perceptron(self):
        """Creates a perceptron for use in tests."""
        settings = PerceptronSettings(self.weights, self.learning_rate,
                                      self.iterations, 0.01)
        return Perceptron(settings, self.train_points, lambda x: x)

    def test_constructor_learning_rate_set(self):
        """Check learning rate is set properly in constructor."""
        perceptron = self.create_perceptron()
        self.assertEqual(perceptron._Perceptron__learning_rate,
                         self.learning_rate)

    def test_constructor_weights_set(self):
        """Check weights are set properly in constructor."""
        perceptron_weights = self.create_perceptron()._Perceptron__weights
        self.assertTrue((perceptron_weights == self.weights).all())

    def test_constructor_train_set(self):
        """Check training points are set properly in constructor."""
        perceptron_train = self.create_perceptron()._Perceptron__train_points
        self.assertTrue(
            (perceptron_train[0] == self.formatted_training_points[0]).all() and
            (perceptron_train[1] == self.formatted_training_points[1]).all())

    def test_format_training_points(self):
        """Check if the incoming training points are formatted correctly for
        classification.
        """
        fomatted_points = Perceptron.format_training_points(self.train_points)
        self.assertTrue(
            (fomatted_points[0] == self.formatted_training_points[0]).all() and
            (fomatted_points[1] == self.formatted_training_points[1]).all())

    def test_final_weights(self):
        """Check if correct weights are produced."""
        perceptron = self.create_perceptron()
        perceptron.run_perceptron()
        self.assertTrue(np.isclose(perceptron._Perceptron__weights,
                                   self.final_weights).all())


if __name__ == "__main__":
    unittest.main()
