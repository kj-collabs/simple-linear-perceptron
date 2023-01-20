import numpy as np

class Perceptron(object):
    def __init__(self):
        self.__w = np.array([1.0, -1.0, 1.0])
        self.__p = 0.5
        self.__train = np.array([
            [[0, 1, 1], [1, 0, 1]],
            [[1, 3, 1], [3, 0, 1]]
        ])

    def set_learning_rate(self, p):
        self.__p = p

    def set_train_data(self, train_data):
        self.__train = train_data

    def set_weights(self, w):
        self.__w = w

    def run_perceptron(self):
        class1_points, class1 = self.__train[0], 1
        class2_points, class2 = self.__train[1], -1
        finished = False
        print("here")

        while not finished:
            misclassified_points = []
            for positive_point in class1_points:
                wx = self.__w@positive_point
                if (class2 * wx) >= 0:
                    print(f"Misclassified {positive_point} as class 2. Output: {wx}")
                    misclassified_points.append(positive_point * class2)

            for negative_point in class2_points:
                wx = self.__w@negative_point
                if (class1 * wx) >= 0:
                    print(f"Misclassified {negative_point} as class 1. Output: {wx}")
                    misclassified_points.append(negative_point * class1)

            if len(misclassified_points) == 0:
                finished = True

            print(np.sum(np.array(misclassified_points), axis=0))
            self.__w = self.__w - self.__p * np.sum(np.array(misclassified_points), axis=0)

            print(f"Updated weights. New weights: {self.__w}")

if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.run_perceptron()
