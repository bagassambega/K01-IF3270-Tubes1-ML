from typing import Callable
import numpy as np

class Perceptron:
    """
    Implementation of one Perceptron
    """
    def __init__(self, weights: np.ndarray, activation_function: Callable, inputs: np.ndarray):
        """
        Create a perceptron. Receives matrix of weights (also with the bias), activation function, and matrix of inputs (x0 included)
        """
        self.activation_function = activation_function
        self.weights = weights
        self.inputs = inputs

    def net(self) -> float:
        """
        Calculate net/sigma by multiplying each input and its weight, and sum all of them
        """
        return float(np.dot(self.weights.T, self.inputs))

    def activate(self, x: float) -> float:
        """
        Called activation function
        """
        return self.activation_function(x)
